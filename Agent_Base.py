import os
import sys
import logging
import json
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
        model: str = "qwen3-235b-a22b-instruct-2507",
        gpus: list = [0, 1],
    ):
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
            self.logger, result_dir, target_env_path=nnunet_env_path
        )
        self.reflection = ReflectionModule(self.client, result_dir, model, self.logger)

        # 5. Config
        self.target_organs = [t.strip() for t in target_organs.split(",")]
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

    def run(
        self,
        run_inference: bool = True,
        run_Knowledge_Distillation: bool = True,
        run_reflection: bool = True,
    ):
        """
        Execute the Agentic Workflow.
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Medical SFUDA Agent")
        self.logger.info(f"Target Organs: {self.target_organs}")
        self.logger.info(f"Data Path: {self.target_data_path}")
        self.logger.info("=" * 50)

        json_path = None
        constructed_dataset_path = None

        # --- Phase 1 & 2: Perception & Planning ---
        self.logger.info("\n>>> Phase 1 & 2: Perception & Planning")
        try:
            samples, data_info = self.perception.scan_and_sample(self.target_data_path)
            domain_context = self.perception.analyze_domain(self.target_organs, samples)

            # Construct prompts
            self.perception.construct_prompts(
                domain_context.get("corrected_targets", self.target_organs),
                domain_context,
            )

            # Save simplified plan (no execution_strategy hardcoded)
            json_path = self.perception.save_adaptation_plan(domain_context, data_info)

            # Load plan for later use
            with open(json_path, "r") as f:
                adaptation_plan = json.load(f)

            self.logger.info(f"Plan generated: {json_path}")

        except Exception as e:
            self.logger.error(f"Perception phase failed: {e}")
            return

        # --- Phase 3 & 4: Action (Execution) ---
        if run_inference:
            self.logger.info("\n>>> Phase 3 & 4: Action (Execution)")
            try:
                constructed_dataset_path = self.action.execute_adaptation_plan(
                    dataset_path=self.target_data_path,
                    adaptation_plan=adaptation_plan,
                    primary_tool=None,
                )

                self.logger.info(
                    f"Action completed. Dataset ready at: {constructed_dataset_path}"
                )

            except Exception as e:
                self.logger.error(f"Action phase failed: {e}")
                import traceback

                self.logger.error(traceback.format_exc())
                return

        # --- Phase 4.5: Reflection ---
        if run_inference and run_reflection and constructed_dataset_path:
            self.logger.info("\n>>> Phase 4.5: Reflection & Quality Control")
            try:
                reflection_summary = self.reflection.reflect_on_dataset(
                    constructed_dataset_path=constructed_dataset_path,
                    adaptation_plan_path=json_path,
                    num_slices_per_case=4,
                )
                # Check for high rejection
                if reflection_summary["rejection_rate"] > 0.4:
                    self.logger.warning(
                        f"High rejection rate ({reflection_summary['rejection_rate']:.1%}). Knowledge distillation might be unstable."
                    )

            except Exception as e:
                self.logger.error(f"Reflection phase failed: {e}")

        if run_inference and run_reflection and constructed_dataset_path:
            self.logger.info("\n>>> Phase 4.6: Iterative Refinement Loop")

            max_refinement_iterations = 3
            refinement_iteration = 0

            while refinement_iteration < max_refinement_iterations:
                with open(json_path, "r") as f:
                    current_plan = json.load(f)

                refinement_strategy = current_plan.get("current_refinement_plan", {})
                stop_refinement = refinement_strategy.get("stop_refinement", True)

                if stop_refinement:
                    self.logger.info(
                        f"Refinement stopping condition met: {refinement_strategy.get('reason', 'Quality threshold reached')}"
                    )

                    self._finalize_reflection_after_refinement(
                        json_path, constructed_dataset_path
                    )
                    break

                refinement_iteration += 1
                self.logger.info(
                    f"\n=== Refinement Iteration {refinement_iteration}/{max_refinement_iterations} ==="
                )

                # Execute refinement
                try:
                    self.action.execute_refinement_actions(
                        constructed_dataset_path=constructed_dataset_path,
                        refinement_strategy=refinement_strategy,
                        adaptation_plan=current_plan,
                    )
                except Exception as e:
                    self.logger.error(f"Refinement execution failed: {e}")
                    import traceback

                    self.logger.error(traceback.format_exc())
                    break

                # Re-reflection with iteration number
                try:
                    affected_cases = self._get_affected_cases(refinement_strategy)

                    if affected_cases:
                        self.logger.info(
                            f"Re-reflecting {len(affected_cases)} refined cases..."
                        )

                        re_reflection_summary = self.reflection.reflect_specific_cases(
                            constructed_dataset_path=constructed_dataset_path,
                            case_names=affected_cases,
                            target_organs=current_plan.get("target_organs", []),
                            organ_label_mapping=self._get_organ_label_mapping(
                                constructed_dataset_path
                            ),
                            iteration=refinement_iteration,
                        )

                        self.reflection.merge_reflection_results(
                            adaptation_plan_path=json_path,
                            new_results=re_reflection_summary,
                        )

                except Exception as e:
                    self.logger.error(f"Re-reflection failed: {e}")
                    import traceback

                    self.logger.error(traceback.format_exc())
                    break

            if refinement_iteration >= max_refinement_iterations:
                self.logger.warning(
                    "Reached maximum refinement iterations without convergence."
                )
                self._finalize_reflection_after_refinement(
                    json_path, constructed_dataset_path
                )

        # --- Phase 5: Knowledge_Distillation ---
        if run_inference and run_Knowledge_Distillation and constructed_dataset_path:
            self.logger.info("\n>>> Phase 5: Knowledge_Distillation")
            try:
                dataset_id = self.finetuning.extract_dataset_id(
                    constructed_dataset_path
                )

                # Auto-configure and Preprocess
                auto_config = self.finetuning.run_preprocessing(dataset_id)

                self.finetuning.run_training(
                    dataset_id, auto_config, pretrained_weights=None
                )

            except Exception as e:
                self.logger.error(f"Knowledge_Distillation phase failed: {e}")

        self.logger.info("\n" + "=" * 50)
        self.logger.info("Workflow Finished.")
        self.logger.info("=" * 50)

    def _get_affected_cases(self, refinement_strategy: Dict) -> List[str]:
        """
        Extract all affected case names from refinement strategy
        """
        affected = set()
        actions = refinement_strategy.get("actions", {})

        for tool_name, tool_config in actions.items():
            cases = tool_config.get("cases", [])
            affected.update(cases)

        return list(affected)

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

    def _finalize_reflection_after_refinement(
        self, json_path: str, constructed_dataset_path: str
    ):
        """
        After refinement loop ends, update reflection with final state.
        Clean up rejected cases and update dataset.json.
        Add final status description.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINALIZING REFLECTION RESULTS")
        self.logger.info("=" * 60)

        with open(json_path, "r") as f:
            final_plan = json.load(f)

        latest_reflection = final_plan.get("reflection", {})
        final_rejected_cases = set(latest_reflection.get("rejected_cases", []))

        images_dir = os.path.join(constructed_dataset_path, "imagesTr")
        labels_dir = os.path.join(constructed_dataset_path, "labelsTr")

        # Remove rejected cases
        if final_rejected_cases:
            self.logger.info(
                f"Removing {len(final_rejected_cases)} rejected cases from dataset..."
            )

            self.reflection._remove_rejected_cases(
                list(final_rejected_cases), images_dir, labels_dir
            )

            remaining_count = latest_reflection["total_evaluated"] - len(
                final_rejected_cases
            )
            self.logger.info(
                f"Removed {len(final_rejected_cases)} cases, {remaining_count} remaining"
            )
        else:
            remaining_count = latest_reflection.get("total_evaluated", 0)
            self.logger.info(
                f"No rejected cases to remove, {remaining_count} cases in dataset"
            )

        self.reflection._update_dataset_json(constructed_dataset_path, remaining_count)

        with open(json_path, "r") as f:
            final_plan = json.load(f)

        refinement_strategy = final_plan.get("current_refinement_plan", {})

        # Add final status description in reflection
        if "reflection" in final_plan:
            final_plan["reflection"]["final_status"] = {
                "refinement_completed": True,
                "stop_reason": refinement_strategy.get(
                    "reason", "Refinement loop completed"
                ),
                "final_dataset_size": remaining_count,
                "final_rejected_count": len(final_rejected_cases),
                "finalized_at": datetime.now().isoformat(),
            }

            with open(json_path, "w") as f:
                json.dump(final_plan, f, indent=2)

        self.logger.info(
            f"Final dataset: {remaining_count} cases (reflection finalized)"
        )
        self.logger.info(
            f"Refinement reason: {refinement_strategy.get('reason', 'N/A')}"
        )
        self.logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    # Config
    API_KEY = ""
    BASE_URL = ""
    TARGET_DATA = ""
    TARGETS = "Spleen,Right Kidney,Left Kidney,Gallbladder,Esophagus,Liver,Stomach,Aortic,Vena Cava,Pancreas,Right Adrenal,Left Adrenal,Duodenum,Bladder,Prostate/Uterus"

    BP_CKPT = ""
    SAM3_CKPT = ""

    RESULT_DIR = "./Agent-Result"

    agent = MedicalSFUDAAgent(
        target_organs=TARGETS,
        target_data_path=TARGET_DATA,
        result_dir=RESULT_DIR,
        api_key=API_KEY,
        base_url=BASE_URL,
        biomedparse_ckpt=BP_CKPT,
        sam3_ckpt=SAM3_CKPT,
        gpus=[0, 1],
    )

    agent.run(run_inference=True, run_reflection=True, run_Knowledge_Distillation=False)
