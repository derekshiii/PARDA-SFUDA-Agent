import os
import sys
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from Perception import PerceptionModule
from Action import ActionModule
from Reflection import ReflectionModule
from Knowledge_Distillation import FinetuningModule
from typing import Dict, List

BiomedParse_DIR = "./BiomedParse"
if BiomedParse_DIR not in sys.path:
    sys.path.append(BiomedParse_DIR)

SAM3_DIR = "./sam3"
if SAM3_DIR not in sys.path:
    sys.path.append(SAM3_DIR)


class MedicalSFUDAAgent:
    def __init__(
        self,
        target_organs: str,
        target_data_path: str,
        result_dir: str,
        api_key: str,
        base_url: str,
        # Checkpoints
        biomedparse_ckpt: str = "./BiomedParse/biomedparse_3D_AllData_MultiView_edge.ckpt",
        sam3_ckpt: str = "./sam3/sam3/sam3.pt",
        nnunet_env_path: str = None,
        model: str = "Qwen3-VL-32B-Instruct",
        gpus: list = None,
        source_checkpoint: str = None,
    ):
        gpus = [0] if gpus is None else list(gpus)
        self.source_checkpoint = source_checkpoint
        self.max_refinement_iterations = 5
        # 1. Setup Directories
        self.result_dir = result_dir
        self.log_dir = os.path.join(result_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 2. Setup Logging
        self.logger = self._setup_logger()

        # 3. Setup Client
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 4. Initialize Sub-modules
        self.perception = PerceptionModule(self.client, result_dir, model, self.logger)

        self.action = ActionModule(
            result_dir=result_dir,
            logger=self.logger,
            biomedparse_checkpoint=biomedparse_ckpt,
            sam3_checkpoint=sam3_ckpt,
            gpus=gpus,
        )

        self.finetuning = FinetuningModule(
            self.logger, result_dir, target_env_path=nnunet_env_path, gpus=gpus
        )
        self.reflection = ReflectionModule(self.client, result_dir, model, self.logger)

        # 5. Config
        self.target_organs = [t.strip() for t in target_organs.split(",")]
        if any(not organ for organ in self.target_organs):
            raise ValueError("Target organ names must be non-empty")
        self.target_data_path = target_data_path

    def _setup_logger(self):
        """Configure logging to file and console."""
        logger = logging.getLogger("SFUDA_Agent")
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear existing

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(self.log_dir, f"agent_run_{timestamp}.log")
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def run(self, run_inference=True, run_Knowledge_Distillation=True, run_reflection=True):
        if run_Knowledge_Distillation and not run_reflection:
            raise ValueError("Paper distillation requires Reflection-curated ACCEPT samples")
        if run_Knowledge_Distillation and (not self.source_checkpoint or not Path(self.source_checkpoint).is_file()):
            raise ValueError("Distillation requires an existing source_checkpoint trained for 500 epochs")
        if not run_inference:
            raise ValueError("Use FinetuningModule for distillation of an existing curated dataset")
        samples, data_info = self.perception.scan_and_sample(self.target_data_path)
        context = self.perception.analyze_domain(self.target_organs, samples)
        self.perception.construct_prompts(context.get("corrected_targets", self.target_organs), context)
        json_path = self.perception.save_adaptation_plan(context, data_info)
        with open(json_path) as stream:
            plan = json.load(stream)
        dataset = self.action.execute_adaptation_plan(self.target_data_path, plan)
        plan.update(source_checkpoint=self.source_checkpoint, refinement_budget=self.max_refinement_iterations,
                    vlm_model=self.perception.model, constructed_dataset_path=dataset)
        with open(json_path, "w") as stream:
            json.dump(plan, stream, indent=2)
        if run_reflection:
            self.reflection.reflect_on_dataset(dataset, json_path, num_slices_per_case=4)
            self._run_refinement(json_path, dataset)
        if run_Knowledge_Distillation:
            info = json.loads((Path(dataset) / "dataset.json").read_text())
            if not info["numTraining"]:
                raise ValueError("No ACCEPT samples remain for distillation")
            dataset_id = self.finetuning.extract_dataset_id(dataset)
            configuration = self.finetuning.run_preprocessing(dataset_id, self.source_checkpoint)
            self.finetuning.run_training(dataset_id, configuration, checkpoint_path=self.source_checkpoint)
        self.logger.info(f"Workflow finished: {dataset}")
        return dataset

    def _run_refinement(self, json_path, dataset):
        attempts = {}
        completed = False
        try:
            for iteration in range(1, self.max_refinement_iterations + 1):
                with open(json_path) as stream:
                    plan = json.load(stream)
                reflection = plan["reflection"]
                summary = self.reflection._summarize_evaluations(reflection.get("detailed_evaluations", []))
                reflection.update(summary)
                pending = [case for case in summary["rejected_cases"]
                           if attempts.get(case, 0) < self.max_refinement_iterations]
                if not pending:
                    break
                subset = self.reflection._summarize_evaluations([
                    ev for ev in summary["detailed_evaluations"] if ev["case_name"] in pending])
                strategy = self.reflection.generate_refinement_plan(subset)
                outcomes = self.action.execute_refinement_actions(dataset, strategy, plan)
                for case in pending:
                    attempts[case] = attempts.get(case, 0) + 1
                    outcomes.setdefault(case, {"status": "failed", "error": "No valid VLM remedy available"})
                results = self.reflection.reflect_specific_cases(dataset, pending,
                    plan["target_organs"], self._get_organ_label_mapping(dataset), iteration=iteration)
                results["tool_results"] = outcomes
                results["attempts"] = dict(attempts)
                self.reflection.merge_reflection_results(json_path, results)
                self.logger.info(f"Refinement {iteration}/{self.max_refinement_iterations}: reassessed {len(pending)} rejected samples")
            completed = True
        finally:
            self._finalize_reflection_after_refinement(json_path, dataset, attempts, refinement_completed=completed)

    def _get_organ_label_mapping(self, dataset_path: str) -> Dict[str, int]:
        """Read organ-label mapping from dataset.json"""
        dataset_json_path = os.path.join(dataset_path, "dataset.json")
        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)

        organ_label_mapping = {}
        labels_dict = dataset_info.get("labels", {})
        for organ_name, label_value in labels_dict.items():
            if organ_name != "background":
                organ_label_mapping[organ_name] = label_value

        return organ_label_mapping

    def _finalize_reflection_after_refinement(self, json_path, constructed_dataset_path, attempts=None, refinement_completed=True):
        dataset = Path(constructed_dataset_path)
        with open(json_path) as stream:
            plan = json.load(stream)
        latest = plan.get("reflection", {})
        latest.update(self.reflection._summarize_evaluations(latest.get("detailed_evaluations", [])))
        latest.pop("uncertain", None)
        latest.pop("uncertain_cases", None)
        accepted = set(latest["accepted_cases"])
        info = json.loads((dataset / "dataset.json").read_text())
        ending = info["file_ending"]
        retained, excluded, moves = [], [], []
        archive = Path(self.result_dir) / "excluded_cases" / dataset.name
        for image in sorted((dataset / "imagesTr").glob("*_0000" + ending)):
            case = image.name[:-len("_0000" + ending)]
            label = dataset / "labelsTr" / f"{case}{ending}"
            if case in accepted and label.is_file():
                retained.append(case)
                continue
            excluded.append(case)
            for folder, path in (("imagesTr", image), ("labelsTr", label)):
                if path.exists():
                    destination = archive / folder / path.name
                    if destination.exists():
                        raise FileExistsError(f"Excluded sample already archived: {destination}")
                    moves.append((path, destination))
        # Check all archive collisions before moving either half of any image-label pair.
        for path, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), destination)
        self.reflection._update_dataset_json(constructed_dataset_path, len(retained))
        latest["final_status"] = {"refinement_completed": refinement_completed, "budget": self.max_refinement_iterations,
            "attempts": attempts or {}, "retained_cases": retained, "excluded_cases": excluded,
            "final_dataset_size": len(retained), "archive": str(archive), "finalized_at": datetime.now().isoformat()}
        plan["reflection"] = latest
        if not refinement_completed:
            reason = "Assessment interrupted; unresolved samples excluded"
        elif not excluded:
            reason = "All samples accepted"
        elif all((attempts or {}).get(case, 0) >= self.max_refinement_iterations for case in excluded):
            reason = "Refinement budget exhausted; rejected samples excluded"
        else:
            reason = "Refinement ended; unaccepted or unassessed samples excluded"
        plan["current_refinement_plan"] = {"generated_at": datetime.now().isoformat(), "actions": {},
                                           "failed_cases": [], "stop_refinement": True, "reason": reason}
        with open(json_path, "w") as stream:
            json.dump(plan, stream, indent=2)
        self.logger.info(f"Curated dataset: {len(retained)} ACCEPT, {len(excluded)} excluded")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PARDA paper adaptation pipeline")
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--targets", required=True, help="Comma-separated organ names")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default="Qwen3-VL-32B-Instruct")
    parser.add_argument("--biomedparse-checkpoint", required=True)
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--nnunet-env")
    parser.add_argument("--result-dir", default="./Agent-Result")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--skip-distillation", action="store_true")
    args = parser.parse_args()
    agent = MedicalSFUDAAgent(target_organs=args.targets, target_data_path=args.target_data,
        result_dir=args.result_dir, api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url,
        biomedparse_ckpt=args.biomedparse_checkpoint, sam3_ckpt=args.sam3_checkpoint,
        nnunet_env_path=args.nnunet_env, model=args.model, gpus=[int(g) for g in args.gpus.split(",")],
        source_checkpoint=args.source_checkpoint)
    agent.run(run_Knowledge_Distillation=not args.skip_distillation)
