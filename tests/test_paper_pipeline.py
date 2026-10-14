"""Paper contract checks; model inference is replaced with deterministic masks."""
import json
import logging
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import nibabel as nib
import numpy as np
from PIL import Image

from Action import ActionModule
from Agent_Base import MedicalSFUDAAgent
from Knowledge_Distillation import FinetuningModule
from Perception import PerceptionModule
from Prompts import PROMPT_REGISTRY
from Reflection import ReflectionModule
from Toolbox import BiomedParse2DPredictor, CLAHE, TopologyCleaner, ZoomSAM3Refiner
from nnUNet2BiomedParse import load_volume, process_single_file, save_volume
from run_sam3_standalone import StandaloneSAM3Predictor

LOGGER = logging.getLogger('paper_tests')
LOGGER.addHandler(logging.NullHandler())
CRITERIA = ('anatomical_plausibility', 'boundary_precision', 'morphological_consistency', 'false_positives')


def critic(decision='REJECT', tool='TopologyCleaner'):
    return {'final_decision': decision, 'confidence': 0.9,
            'reasoning': {k: 'Observed image evidence' for k in CRITERIA},
            'scores': {k: 8 for k in CRITERIA},
            'suggested_remedy': {'tool': tool if decision != 'ACCEPT' else 'None', 'rationale': 'Observed fragments'}}


class PaperPipelineTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='parda_test_')
        self.root = Path(self.work.name)
        self.affine = np.diag([1.4, 1.8, 3.0, 1.0])
        self.images = np.arange(24 * 24 * 3, dtype=np.float32).reshape(24, 24, 3)
        self.masks = np.zeros((24, 24, 3), dtype=np.uint16)
        self.masks[3:8, 3:8, :] = 1
        self.masks[18, 18, 1] = 1

    def tearDown(self):
        self.work.cleanup()

    def dataset(self, name='Dataset123_Input', cases=('a',), png=False, labels=True, organs=None):
        path = self.root / name
        (path / 'imagesTr').mkdir(parents=True)
        if labels:
            (path / 'labelsTr').mkdir()
        ending = '.png' if png else '.nii.gz'
        for case in cases:
            if png:
                rgb = np.zeros((24, 24, 3), dtype=np.uint8)
                rgb[..., 0] = np.arange(24, dtype=np.uint8)[:, None] * 10
                rgb[..., 1] = 130
                Image.fromarray(rgb).save(path / 'imagesTr' / f'{case}_0000.png')
                if labels:
                    Image.fromarray(self.masks[..., 1].astype(np.uint8)).save(path / 'labelsTr' / f'{case}.png')
            else:
                nib.save(nib.Nifti1Image(self.images, self.affine), path / 'imagesTr' / f'{case}_0000.nii.gz')
                if labels:
                    nib.save(nib.Nifti1Image(self.masks, self.affine), path / 'labelsTr' / f'{case}.nii.gz')
        (path / 'dataset.json').write_text(json.dumps({'file_ending': ending, 'numTraining': len(cases),
            'channel_names': {'0': 'R', '1': 'G', '2': 'B'} if png else {'0': 'CT'},
            'labels': organs or {'background': 0, 'Liver': 1}}))
        return path

    def action(self):
        action = ActionModule.__new__(ActionModule)
        action.logger, action.gpus = LOGGER, [0]
        action.result_dir = str(self.root)
        action.temp_dir = str(self.root / 'temp')
        action.temp_root = action.temp_dir
        Path(action.temp_dir).mkdir(exist_ok=True)
        action.pseudolabel_dir = str(self.root / 'pseudo')
        Path(action.pseudolabel_dir).mkdir(exist_ok=True)
        action.biomed_input_dir = str(self.root / 'npz')
        action.sam3_checkpoint = 'test-checkpoint'
        action.sam3_wrapper = SimpleNamespace(predict=self.fake_sam)
        return action

    def fake_sam(self, config_json_path, output_dir, **kwargs):
        config = json.loads(Path(config_json_path).read_text())
        for case, entry in config.items():
            reference = load_volume(entry['image'])
            mask = np.zeros(reference.shape, dtype=np.uint16)
            for prompt in entry['prompts']:
                obj_id = prompt['obj_id']
                if obj_id == 2:
                    mask[14:19, 3:8, :] = 2
                else:
                    mask[1:-1, 1:-1, :] = obj_id
            save_volume(mask, reference, Path(output_dir) / f'{case}.nii.gz')

    def test_backend_environment_reaches_python_and_checkpoint_selection(self):
        backend = self.root / 'RGB backend'
        settings = {'BIOMEDPARSE2D_ROOT': str(backend),
                    'BIOMEDPARSE2D_PYTHON': str(self.root / 'RGB environment' / 'bin' / 'python'),
                    'BIOMEDPARSE2D_CHECKPOINT': str(self.root / 'custom weights.pt'),
                    'SAM3_ROOT': str(self.root / 'SAM3 backend'),
                    'SAM3_PYTHON': str(self.root / 'SAM3 environment' / 'bin' / 'python')}
        with patch.dict('os.environ', settings):
            with patch('Toolbox.subprocess.run') as execute:
                BiomedParse2DPredictor().predict('input images', 'output masks', {'Liver': 'liver'}, [2])
            command = execute.call_args.args[0]
            self.assertEqual(command[0], settings['BIOMEDPARSE2D_PYTHON'])
            self.assertEqual(command[command.index('--checkpoint') + 1], settings['BIOMEDPARSE2D_CHECKPOINT'])
            self.assertEqual(command[command.index('--gpu') + 1], '2')
            self.assertTrue(execute.call_args.kwargs['check'])
            child = subprocess.run([sys.executable, '-c',
                'import json; from Action import SAM3_PYTHON_PATH; from run_sam3_standalone import SAM3_DIR; '
                'print(json.dumps([SAM3_PYTHON_PATH, SAM3_DIR]))'], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(child.stdout), [settings['SAM3_PYTHON'], settings['SAM3_ROOT']])

    def test_unlabeled_conversion_ignores_gt_and_preserves_geometry(self):
        path = self.dataset(labels=False)
        out = self.root / 'converted'
        out.mkdir()
        message, success = process_single_file((path / 'imagesTr' / 'a_0000.nii.gz', path / 'labelsTr', out, {'1': 'Liver'}))
        self.assertTrue(success, message)
        with np.load(out / 'a.npz', allow_pickle=True) as data:
            self.assertNotIn('gts', data)
            np.testing.assert_array_equal(data['affine'], load_volume(path / 'imagesTr' / 'a_0000.nii.gz').affine)
            self.assertEqual(len(data['header'].tobytes()), 348)
            nib.Nifti1Header(binaryblock=data['header'].tobytes())
        (path / 'labelsTr').mkdir()
        (path / 'labelsTr' / 'a.nii.gz').write_bytes(b'invalid GT is never read')
        self.assertTrue(process_single_file((path / 'imagesTr' / 'a_0000.nii.gz', path / 'labelsTr', out, {'1': 'Liver'}))[1])

    def test_strategy_clahe_and_original_training_image(self):
        source = self.dataset(labels=False)
        perception = PerceptionModule(None, str(self.root), 'model', LOGGER)
        context = {'corrected_targets': ['Liver'], 'primary_tool': 'BiomedParse', 'modality': 'CT',
                   'anatomical_site': 'Abdomen', 'image_characteristics': {'contrast_rating': 2},
                   'strategy_recommendations': {}}
        perception.construct_prompts(['Liver'], context)
        plan = json.loads(Path(perception.save_adaptation_plan(context, {'input_dimension': '3D'})).read_text())
        action = self.action()
        observed = []

        def predict(data_path, output_dir, **kwargs):
            with np.load(Path(data_path) / 'a.npz', allow_pickle=True) as data:
                observed.append(data['imgs'].copy())
                nib.save(nib.Nifti1Image(self.masks, data['affine']), Path(output_dir) / 'a.nii.gz')
            return output_dir

        action.bp_predictor = SimpleNamespace(predict=predict)
        with patch.dict('os.environ', {'nnUNet_raw': str(self.root)}):
            output = Path(action.execute_adaptation_plan(str(source), plan))
        self.assertEqual(plan['strategy_recommendations']['primary_tool'], 'BiomedParse')
        self.assertNotEqual(plan['working_dataset_path'], str(source))
        self.assertTrue(observed)
        self.assertEqual((output / 'imagesTr' / 'a_0000.nii.gz').read_bytes(), (source / 'imagesTr' / 'a_0000.nii.gz').read_bytes())
        np.testing.assert_allclose(load_volume(output / 'labelsTr' / 'a.nii.gz').affine, self.affine)
        samples, info = perception.scan_and_sample(str(output))
        self.assertEqual(info['total_files'], 1)
        self.assertTrue(all('_0000' in sample['file'] for sample in samples))

    def test_binary_scores_and_low_rejection_rate(self):
        result = ReflectionModule._validate_critic_result(critic('UNCERTAIN'))
        self.assertEqual(result['final_decision'], 'REJECT')
        invalid = critic(); invalid['scores']['boundary_precision'] = 11
        with self.assertRaises(ValueError):
            ReflectionModule._validate_critic_result(invalid)
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        evs = [{'case_name': str(i), 'vlm_evaluation': critic('ACCEPT' if i else 'UNCERTAIN')} for i in range(100)]
        summary = reflection._summarize_evaluations(evs)
        self.assertEqual(summary['rejected'], 1)
        self.assertNotIn('uncertain', summary)
        plan = reflection.generate_refinement_plan(summary)
        self.assertFalse(plan['stop_refinement'])
        self.assertEqual(plan['actions']['TopologyCleaner']['cases'], ['0'])

    def test_prompt_examples_can_be_returned_as_valid_responses(self):
        perception_prompt = PROMPT_REGISTRY['PERCEPTION_ANALYSIS']
        example = json.loads('{' + perception_prompt.split('### Output Format:', 1)[1].split('\n{', 1)[1])
        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
            (calls.append(kw), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(example)))]))[1])))
        perception = PerceptionModule(client, str(self.root), 'model', LOGGER)
        result = perception.analyze_domain(example['corrected_targets'], [{'image_base64': ''}])
        self.assertEqual(result['primary_tool'], example['primary_tool'])
        self.assertEqual(calls[0]['messages'][0]['content'], perception_prompt)
        reflection_prompt = PROMPT_REGISTRY['REFLECTION_CRITIC']
        example = json.loads('{' + reflection_prompt.split('Output ONLY valid JSON:', 1)[1].split('\n{', 1)[1])
        result = ReflectionModule._validate_critic_result(example)
        self.assertEqual(result['final_decision'], 'REJECT')
        self.assertEqual(set(result['scores']), set(CRITERIA))
        self.assertIn('Never output UNCERTAIN', reflection_prompt)

    def test_refinement_stop_uses_case_decisions_instead_of_stale_counts(self):
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        evaluation = {'case_name': 'a', 'vlm_evaluation': critic('UNCERTAIN', 'Zoom')}
        plan = reflection.generate_refinement_plan({'detailed_evaluations': [evaluation]})
        self.assertFalse(plan['stop_refinement'])
        self.assertEqual(plan['actions']['ZoomSAM3']['cases'], ['a'])
        evaluation['vlm_evaluation'] = critic('ACCEPT')
        plan = reflection.generate_refinement_plan({'detailed_evaluations': [evaluation], 'uncertain_cases': ['a']})
        self.assertTrue(plan['stop_refinement'])
        plan = reflection.generate_refinement_plan({'uncertain_cases': ['unassessed']})
        self.assertFalse(plan['stop_refinement'])
        self.assertEqual(plan['failed_cases'], ['unassessed'])

    def test_reject_requires_explanation_for_remedy_and_retries_bad_responses(self):
        bad = critic('UNCERTAIN'); bad['suggested_remedy']['rationale'] = ''
        with self.assertRaises(ValueError):
            ReflectionModule._validate_critic_result(bad)
        invalids = []
        for field in ('scores', 'reasoning', 'suggested_remedy'):
            bad = critic('ACCEPT'); bad[field] = None; invalids.append(bad)
        for score in (0, 11, True, 8.5):
            bad = critic('ACCEPT'); bad['scores']['boundary_precision'] = score; invalids.append(bad)
        bad = critic('ACCEPT or REJECT'); invalids.append(bad)
        bad = critic('UNCERTAIN', 'None'); invalids.append(bad)
        for invalid in invalids:
            with self.subTest(invalid=invalid):
                calls = []
                responses = iter((invalid, critic('UNCERTAIN', 'Zoom')))
                def respond(**kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))])
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
                reflection = ReflectionModule(client, str(self.root), 'model', LOGGER)
                result = reflection._call_vlm_critic(['Liver'], {'Liver': 1}, [0], '', 'a', retries=1, retry_backoff_sec=0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result['final_decision'], 'REJECT')
                self.assertEqual(result['suggested_remedy']['tool'], 'ZoomSAM3')

    def test_critic_uses_actual_label_ids_and_rejects_invalid_masks(self):
        path = self.dataset(organs={'background': 0, 'Kidney': 1, 'Liver': 2})
        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
            (calls.append(kw), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(critic('ACCEPT'))))]))[1])))
        reflection = ReflectionModule(client, str(self.root), 'model', LOGGER)
        mapping = {'Liver': 2, 'Kidney': 1}
        evaluation = reflection.evaluate_case(str(path / 'imagesTr' / 'a_0000.nii.gz'), str(path / 'labelsTr' / 'a.nii.gz'),
                                              ['Liver', 'Kidney'], mapping)
        self.assertEqual(evaluation['vlm_evaluation']['final_decision'], 'ACCEPT')
        prompt = calls[0]['messages'][1]['content'][0]['text']
        self.assertIn('label 2 = Liver, contour RGB (30, 160, 60)', prompt)
        self.assertIn('label 1 = Kidney, contour RGB (180, 30, 30)', prompt)
        for invalid_value in (1.5, 3, -1, np.nan):
            with self.subTest(value=invalid_value):
                mask = self.masks.astype(np.float32); mask[0, 0, 0] = invalid_value
                nib.save(nib.Nifti1Image(mask, self.affine), path / 'labelsTr' / 'a.nii.gz')
                result = reflection._evaluate_cases(str(path), ['a'], ['Liver', 'Kidney'], mapping)
                self.assertEqual(result[0]['vlm_evaluation']['final_decision'], 'REJECT')
                self.assertIn('error', result[0]['vlm_evaluation'])
        self.assertEqual(len(calls), 1)

    def test_many_organ_contours_remain_distinct_and_match_legends(self):
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        labels = np.zeros((24, 24), dtype=np.int32)
        labels[3:8, 3:8] = 1
        labels[14:19, 14:19] = 11
        overlay = np.asarray(reflection._create_multi_class_overlay(self.images[..., 0], labels))
        self.assertFalse(np.array_equal(overlay[3, 3], overlay[14, 14]))
        palette = reflection._get_boundary_palette(16)
        self.assertEqual(len({tuple(c.astype(np.uint8)) for c in palette.values()}), 16)
        np.testing.assert_array_equal(overlay[14, 14], palette[11].astype(np.uint8))

    def test_single_slice_png_and_technical_failure(self):
        case = 'patient_0000_scan'
        path = self.dataset(png=True, cases=(case,))
        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
            (calls.append(kw), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(critic('ACCEPT'))))]))[1])))
        reflection = ReflectionModule(client, str(self.root), 'model', LOGGER)
        evaluation = reflection.evaluate_case(str(path / 'imagesTr' / f'{case}_0000.png'), str(path / 'labelsTr' / f'{case}.png'), ['Liver'], {'Liver': 1})
        self.assertEqual(evaluation['case_name'], case)
        self.assertEqual(evaluation['evaluated_slices'], 1)
        self.assertEqual(evaluation['vlm_evaluation']['final_decision'], 'ACCEPT')
        self.assertEqual(calls[0]['temperature'], 0.1)
        self.assertEqual(calls[0]['max_tokens'], 4096)
        self.assertIn('label 1 = Liver', calls[0]['messages'][1]['content'][0]['text'])
        with patch.object(client.chat.completions, 'create', side_effect=RuntimeError('offline')):
            failure = reflection._call_vlm_critic(['Liver'], {'Liver': 1}, [0], '', 'a', retries=0)
        self.assertEqual(failure['final_decision'], 'REJECT')
        self.assertIn('error', failure)
        self.assertNotIn('scores', failure)

    def test_topology_uses_per_label_component_counts(self):
        path = self.dataset()
        mask = self.masks.copy(); mask[12:16, 12:16, :] = 1
        label = path / 'labelsTr' / 'a.nii.gz'
        save_volume(mask, load_volume(label), label)
        out = self.root / 'clean.nii.gz'
        TopologyCleaner(LOGGER).run_cleaning(str(label), str(out), {'min_size': 0, 'multi_component_labels': {'1': 2}})
        cleaned = load_volume(out).get_fdata()
        self.assertEqual(cleaned[18, 18, 1], 0)
        self.assertEqual(cleaned[13, 13, 1], 1)
        self.assertEqual(cleaned[5, 5, 1], 1)

    def test_zoom_preserves_unprocessed_labels_and_missing_output_fails(self):
        path = self.dataset()
        label = path / 'labelsTr' / 'a.nii.gz'
        mask = self.masks.copy(); mask[15:20, 3:8, :] = 2
        save_volume(mask, load_volume(label), label)
        refiner = ZoomSAM3Refiner('python', 'script', 'checkpoint', str(self.root / 'zoom'), LOGGER)
        refiner.sam3.predict = self.fake_sam
        out = self.root / 'zoom.nii.gz'
        refiner.run(str(path / 'imagesTr' / 'a_0000.nii.gz'), str(label), str(out), {1: 'Left Kidney'}, margin=2)
        result = load_volume(out).get_fdata()
        np.testing.assert_array_equal(result == 2, mask == 2)
        self.assertFalse(np.array_equal(result == 1, mask == 1))
        refiner.sam3.predict = lambda *args, **kwargs: None
        with self.assertRaises(FileNotFoundError):
            refiner.run(str(path / 'imagesTr' / 'a_0000.nii.gz'), str(label), str(out), {1: 'Left Kidney'}, margin=2)

    def test_zoom_png_crops_preserve_rgb_pixels(self):
        path = self.dataset(png=True)
        image = path / 'imagesTr' / 'a_0000.png'
        coarse = self.root / 'coarse.nii.gz'
        nib.save(load_volume(path / 'labelsTr' / 'a.png'), coarse)
        refiner = ZoomSAM3Refiner('python', 'script', 'checkpoint', str(self.root / 'zoom_rgb'), LOGGER)
        observed = []
        def predict(config_path, output_dir, **kwargs):
            config = json.loads(Path(config_path).read_text())
            for entry in config.values():
                with Image.open(entry['image']) as crop:
                    self.assertEqual(crop.mode, 'RGB')
                    observed.append(np.asarray(crop).copy())
            self.fake_sam(config_path, output_dir, **kwargs)
        refiner.sam3.predict = predict
        refiner.run(str(image), str(coarse), str(self.root / 'refined_rgb.nii.gz'), {1: 'Liver'}, margin=2)
        with Image.open(image) as original:
            np.testing.assert_array_equal(observed[0], np.asarray(original)[1:21, 1:21])

    def test_clahe_and_symmetry_resegment_and_preserve_other_organs(self):
        organs = {'background': 0, 'Left Kidney': 1, 'Right Kidney': 2}
        path = self.dataset(organs=organs)
        action = self.action()
        plan = {'working_dataset_path': str(path), 'strategy_recommendations': {'organ_pairs': [dict(list(organs.items())[1:])]}, 'target_organs': ['Left Kidney', 'Right Kidney']}
        result = action.execute_refinement_actions(str(path), {'actions': {'CLAHE': {'cases': ['a'], 'params': {}}}}, plan)
        self.assertEqual(result['a']['status'], 'success')
        label = path / 'labelsTr' / 'a.nii.gz'
        save_volume(self.masks, load_volume(label), label)
        result = action.execute_refinement_actions(str(path), {'actions': {'SymmetryChecker': {'cases': ['a'], 'params': {}}}}, plan)
        self.assertEqual(result['a']['status'], 'success')
        refined = load_volume(label).get_fdata()
        np.testing.assert_array_equal(refined == 1, self.masks == 1)
        self.assertTrue(np.any(refined == 2))

    def test_tool_failure_keeps_original_and_does_not_stop_other_cases(self):
        path = self.dataset(cases=('a', 'b'))
        original = (path / 'labelsTr' / 'a.nii.gz').read_bytes()
        action = self.action()
        action.sam3_wrapper.predict = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('offline'))
        plan = {'strategy_recommendations': {}, 'target_organs': ['Liver']}
        result = action.execute_refinement_actions(str(path), {'actions': {
            'CLAHE': {'cases': ['a'], 'params': {}}, 'TopologyCleaner': {'cases': ['b'], 'params': {'min_size': 0}}}}, plan)
        self.assertEqual(result['a']['status'], 'failed')
        self.assertEqual(result['b']['status'], 'success')
        self.assertEqual((path / 'labelsTr' / 'a.nii.gz').read_bytes(), original)

    def test_symmetry_preserves_other_healthy_pairs(self):
        organs = {'background': 0, 'Left Kidney': 1, 'Right Kidney': 2, 'Left Adrenal': 3, 'Right Adrenal': 4}
        path = self.dataset(organs=organs)
        label = path / 'labelsTr' / 'a.nii.gz'
        masks = self.masks.copy()
        masks[4:7, 14:17, :] = 3
        masks[17:20, 14:17, :] = 4
        save_volume(masks, load_volume(label), label)
        plan = {'strategy_recommendations': {'organ_pairs': [
            {'Left Kidney': 1, 'Right Kidney': 2}, {'Left Adrenal': 3, 'Right Adrenal': 4}]}}
        results = self.action().execute_refinement_actions(str(path),
            {'actions': {'SymmetryChecker': {'cases': ['a'], 'params': {}}}}, plan)
        self.assertEqual(results['a']['status'], 'success')
        refined = load_volume(label).get_fdata()
        np.testing.assert_array_equal(refined == 3, masks == 3)
        np.testing.assert_array_equal(refined == 4, masks == 4)
        self.assertTrue(np.any(refined == 2))

    def test_refinement_accepts_early_and_excludes_after_five(self):
        path = self.dataset(cases=('a', 'b', 'c'))
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        evaluations = [{'case_name': c, 'vlm_evaluation': critic('ACCEPT' if c == 'a' else 'UNCERTAIN')} for c in ('a', 'b', 'c')]
        plan = {'target_organs': ['Liver'], 'strategy_recommendations': {}, 'working_dataset_path': str(path),
                'reflection': reflection._summarize_evaluations(evaluations)}
        plan_path = self.root / 'plan.json'; plan_path.write_text(json.dumps(plan))
        seen = []

        def reassess(dataset, cases, targets, mapping, iteration=1):
            seen.append((iteration, list(cases)))
            return {'iteration': iteration, 'evaluations': [{'case_name': c, 'vlm_evaluation':
                    critic('ACCEPT' if c == 'b' and iteration == 2 else 'REJECT')} for c in cases]}

        reflection.reflect_specific_cases = reassess
        agent = MedicalSFUDAAgent.__new__(MedicalSFUDAAgent)
        agent.reflection, agent.action = reflection, self.action()
        agent.logger, agent.result_dir, agent.max_refinement_iterations = LOGGER, str(self.root), 5
        agent._run_refinement(str(plan_path), str(path))
        self.assertEqual(seen, [(1, ['b', 'c']), (2, ['b', 'c']), (3, ['c']), (4, ['c']), (5, ['c'])])
        final = json.loads(plan_path.read_text())['reflection']['final_status']
        self.assertEqual(final['attempts'], {'b': 2, 'c': 5})
        self.assertEqual(final['retained_cases'], ['a', 'b'])
        self.assertEqual(json.loads((path / 'dataset.json').read_text())['numTraining'], 2)
        self.assertTrue((Path(final['archive']) / 'labelsTr' / 'c.nii.gz').is_file())
        self.assertFalse((path / 'imagesTr' / 'c_0000.nii.gz').exists())

    def test_legacy_uncertain_is_rejected_in_merged_results_and_final_dataset(self):
        path = self.dataset(cases=('a', 'b', 'c'))
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        evaluations = [{'case_name': c, 'vlm_evaluation': critic('UNCERTAIN')} for c in ('a', 'b', 'c')]
        plan_path = self.root / 'legacy_plan.json'
        plan_path.write_text(json.dumps({'reflection': {'uncertain': 3, 'uncertain_cases': ['a', 'b', 'c'],
            'detailed_evaluations': evaluations}}))
        reflection.merge_reflection_results(str(plan_path), {'iteration': 1,
            'evaluations': [{'case_name': 'a', 'vlm_evaluation': critic('ACCEPT')},
                            {'case_name': 'b', 'vlm_evaluation': critic('UNCERTAIN')}]})
        result = json.loads(plan_path.read_text())
        self.assertEqual(result['reflection']['accepted_cases'], ['a'])
        self.assertEqual(result['reflection']['rejected_cases'], ['b', 'c'])
        self.assertNotIn('uncertain', result['reflection'])
        self.assertEqual(result['reflection_history'][0]['evaluations'][1]['vlm_evaluation']['final_decision'], 'REJECT')
        # Finalization also handles legacy state when no refinement result was merged.
        result['reflection'] = {'uncertain': 2, 'uncertain_cases': ['b', 'c'],
            'detailed_evaluations': [{'case_name': 'a', 'vlm_evaluation': critic('ACCEPT')}, *evaluations[1:]]}
        plan_path.write_text(json.dumps(result))
        agent = MedicalSFUDAAgent.__new__(MedicalSFUDAAgent)
        agent.reflection, agent.logger, agent.result_dir, agent.max_refinement_iterations = reflection, LOGGER, str(self.root), 5
        agent._finalize_reflection_after_refinement(str(plan_path), str(path))
        result = json.loads(plan_path.read_text())['reflection']
        self.assertNotIn('uncertain', result)
        self.assertNotIn('uncertain_cases', result)
        self.assertEqual(result['rejected'], 2)
        self.assertEqual(result['final_status']['retained_cases'], ['a'])
        self.assertEqual(result['final_status']['excluded_cases'], ['b', 'c'])
        self.assertEqual(json.loads((path / 'dataset.json').read_text())['numTraining'], 1)

    def test_archive_collision_does_not_split_image_label_pairs(self):
        path = self.dataset(cases=('a', 'b'))
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        plan_path = self.root / 'collision_plan.json'
        plan_path.write_text(json.dumps({'reflection': reflection._summarize_evaluations([
            {'case_name': c, 'vlm_evaluation': critic('REJECT')} for c in ('a', 'b')])}))
        archive = self.root / 'excluded_cases' / path.name / 'labelsTr'
        archive.mkdir(parents=True)
        (archive / 'b.nii.gz').write_bytes(b'previous archived label')
        agent = MedicalSFUDAAgent.__new__(MedicalSFUDAAgent)
        agent.reflection, agent.logger, agent.result_dir, agent.max_refinement_iterations = reflection, LOGGER, str(self.root), 5
        with self.assertRaises(FileExistsError):
            agent._finalize_reflection_after_refinement(str(plan_path), str(path))
        self.assertEqual(len(list((path / 'imagesTr').iterdir())), 2)
        self.assertEqual(len(list((path / 'labelsTr').iterdir())), 2)
        self.assertEqual((archive / 'b.nii.gz').read_bytes(), b'previous archived label')

    def test_png_dataset_is_constructed_with_original_rgb(self):
        source = self.dataset(labels=False, png=True)
        pseudo = self.root / 'predictions'; pseudo.mkdir()
        ref = load_volume(source / 'imagesTr' / 'a_0000.png')
        save_volume(self.masks[..., 1, None], ref, pseudo / 'a.nii.gz')
        with patch.dict('os.environ', {'nnUNet_raw': str(self.root)}):
            output = Path(self.action()._construct_nnunet_dataset(str(pseudo), str(source), ['Liver'], 'Fundus'))
        info = json.loads((output / 'dataset.json').read_text())
        self.assertEqual(info['file_ending'], '.png')
        self.assertEqual(info['numTraining'], 1)
        self.assertEqual(info['channel_names'], {'0': 'R', '1': 'G', '2': 'B'})
        self.assertEqual((output / 'imagesTr' / 'a_0000.png').read_bytes(), (source / 'imagesTr' / 'a_0000.png').read_bytes())

    def test_png_inference_uses_native_rgb_biomedparse_backend(self):
        source = self.dataset(png=True, labels=False)
        action = self.action()
        calls = []
        def predict_rgb(dataset_path, output_dir, prompts, gpus):
            image = Path(dataset_path) / 'imagesTr' / 'a_0000.png'
            with Image.open(image) as original:
                calls.append((original.mode, prompts, gpus))
            reference = load_volume(image)
            save_volume(self.masks[..., 1, None], reference, Path(output_dir) / 'a.nii.gz')
            return output_dir
        action.bp_2d_predictor = SimpleNamespace(predict=predict_rgb)
        action.bp_predictor = SimpleNamespace(predict=lambda **kwargs: self.fail('RGB input reached the 3D backend'))
        plan = {'target_organs': ['Liver'], 'text_prompts': {'Liver': 'Liver in image'},
                'strategy_recommendations': {'primary_tool': 'BiomedParse'}}
        with patch.dict('os.environ', {'nnUNet_raw': str(self.root)}):
            output = Path(action.execute_adaptation_plan(str(source), plan))
        self.assertEqual(calls, [('RGB', {'Liver': 'Liver in image'}, [0])])
        self.assertEqual((output / 'imagesTr' / 'a_0000.png').read_bytes(), (source / 'imagesTr' / 'a_0000.png').read_bytes())

    def test_training_uses_source_weights_selected_gpu_and_300_epochs(self):
        checkpoint = self.root / 'source.pth'; checkpoint.write_bytes(b'command fixture')
        training = FinetuningModule(LOGGER, str(self.root), gpus=[1])
        calls = []
        training._execute_command = lambda cmd, stage: calls.append(cmd)
        training.run_training(123, '3d_fullres', str(checkpoint))
        cmd = calls[0]
        self.assertTrue(cmd[1].endswith('train_student.py'))
        self.assertEqual(cmd[cmd.index('-pretrained_weights') + 1], str(checkpoint))
        self.assertEqual(cmd[cmd.index('-num_gpus') + 1], '1')
        with self.assertRaises(ValueError):
            training.run_training(123, '3d_fullres', '')
        fake = SimpleNamespace(get_trainer_from_args=lambda *a, **kw: SimpleNamespace(num_epochs=1000))
        with patch.dict('sys.modules', {'nnunetv2': SimpleNamespace(paths=SimpleNamespace()), 'nnunetv2.run': SimpleNamespace(run_training=fake)}):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train_student.py'))
            trainer = namespace['get_paper_trainer']('123', '3d_fullres', 'all')
        self.assertEqual(trainer.num_epochs, 300)

    def test_full_workflow_with_unlabeled_input_and_distillation(self):
        source = self.dataset(labels=False)
        checkpoint = self.root / 'source_wiring_fixture.pth'
        checkpoint.write_bytes(b'Call-wiring fixture; native weight loading is tested separately')
        requests = []
        def respond(**kwargs):
            requests.append(kwargs)
            if 'Senior Radiologist' in kwargs['messages'][0]['content']:
                result = {'corrected_targets': ['Liver'], 'primary_tool': 'BiomedParse',
                          'modality': 'CT', 'anatomical_site': 'Abdomen',
                          'image_characteristics': {'contrast_rating': 2}, 'strategy_recommendations': {}}
            else:
                result = critic('ACCEPT')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
        def predict(_predictor, data_path, output_dir, **kwargs):
            with np.load(Path(data_path) / 'a.npz', allow_pickle=True) as data:
                nib.save(nib.Nifti1Image(self.masks, data['affine']), Path(output_dir) / 'a.nii.gz')
            return output_dir
        with patch('Agent_Base.OpenAI', return_value=client), patch('Action.BiomedParsePredictor.predict', new=predict), \
                patch.dict('os.environ', {'nnUNet_raw': str(self.root)}), \
                patch.object(FinetuningModule, 'run_preprocessing', return_value='3d_fullres') as preprocess, \
                patch.object(FinetuningModule, 'run_training') as train:
            agent = MedicalSFUDAAgent('Liver', str(source), str(self.root / 'results'), 'test', 'http://unused',
                                     source_checkpoint=str(checkpoint))
            output = Path(agent.run())
        dataset_id = agent.finetuning.extract_dataset_id(str(output))
        preprocess.assert_called_once_with(dataset_id, str(checkpoint))
        train.assert_called_once_with(dataset_id, '3d_fullres', checkpoint_path=str(checkpoint))
        self.assertEqual(agent.perception.model, 'Qwen3-VL-32B-Instruct')
        self.assertEqual(agent.action.gpus, [0])
        self.assertEqual(len(requests), 2)
        plan = json.loads(next((self.root / 'results').glob('adaptation_plan_*.json')).read_text())
        self.assertEqual(plan['refinement_budget'], 5)
        self.assertEqual(plan['reflection']['final_status']['retained_cases'], ['a'])
        self.assertEqual((output / 'imagesTr' / 'a_0000.nii.gz').read_bytes(), (source / 'imagesTr' / 'a_0000.nii.gz').read_bytes())

    def test_source_initialization_includes_segmentation_head_and_rejects_mismatch(self):
        import torch
        def network(classes=2):
            model = torch.nn.Module()
            model.encoder = torch.nn.Conv2d(1, 3, 1)
            model.decoder = torch.nn.Module()
            model.decoder.seg_layers = torch.nn.ModuleList([torch.nn.Conv2d(3, classes, 1)])
            return model
        source = network()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(0.25)
        checkpoint = self.root / 'complete_source.pth'
        torch.save({'network_weights': source.state_dict()}, checkpoint)
        fake = SimpleNamespace(get_trainer_from_args=lambda *a, **kw: SimpleNamespace(num_epochs=1000))
        with patch.dict('sys.modules', {'nnunetv2': SimpleNamespace(paths=SimpleNamespace()), 'nnunetv2.run': SimpleNamespace(run_training=fake)}):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train_student.py'))
        student = network()
        namespace['load_source_weights'](student, str(checkpoint))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(student.state_dict()[name], value)
        with self.assertRaises(RuntimeError):
            namespace['load_source_weights'](network(classes=3), str(checkpoint))

    def test_preprocessing_uses_source_configuration_and_architecture(self):
        import torch
        source = self.dataset()
        preprocessed = self.root / 'preprocessed' / source.name
        preprocessed.mkdir(parents=True)
        source_configuration = {'architecture': {'network_class_name': 'ResidualEncoderUNet'},
                                'patch_size': [32, 32], 'batch_size': 2}
        checkpoint = self.root / 'source.pth'
        torch.save({'init_args': {'configuration': '2d', 'dataset_json': {'channel_names': {'0': 'MR'}, 'labels': {'background': '0', 'Liver': '1'}},
                    'plans': {'configurations': {'2d': source_configuration}}}}, checkpoint)
        plans_path = preprocessed / 'nnUNetResEncUNetPlans.json'
        target_configuration = {'architecture': {'network_class_name': 'different'},
                                'patch_size': [16, 16], 'batch_size': 4, 'spacing': [2, 2]}
        training = FinetuningModule(LOGGER, str(self.root))
        def preprocess(cmd, stage):
            self.assertEqual(cmd[cmd.index('-c') + 1], '2d')
            plans_path.write_text(json.dumps({'configurations': {'2d': target_configuration}}))
        training._execute_command = preprocess
        with patch.dict('os.environ', {'nnUNet_raw': str(self.root), 'nnUNet_preprocessed': str(preprocessed.parent)}):
            self.assertEqual(training.run_preprocessing(123, str(checkpoint)), '2d')
        target = json.loads(plans_path.read_text())['configurations']['2d']
        self.assertEqual(target['architecture'], source_configuration['architecture'])
        self.assertEqual(target['patch_size'], [32, 32])
        self.assertEqual(target['spacing'], [2, 2])

    def test_uncertain_api_output_executes_five_refinements_then_excludes(self):
        source = self.dataset(labels=False)
        requests = []
        def respond(**kwargs):
            requests.append(kwargs)
            if 'Senior Radiologist' in kwargs['messages'][0]['content']:
                result = {'corrected_targets': ['Liver'], 'primary_tool': 'BiomedParse', 'modality': 'CT',
                          'anatomical_site': 'Abdomen', 'image_characteristics': {'contrast_rating': 8},
                          'strategy_recommendations': {}}
            else:
                result = critic('UNCERTAIN', 'TopologyCleaner')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
        def predict(_predictor, data_path, output_dir, **kwargs):
            with np.load(Path(data_path) / 'a.npz', allow_pickle=True) as data:
                nib.save(nib.Nifti1Image(self.masks, data['affine']), Path(output_dir) / 'a.nii.gz')
            return output_dir
        with patch('Agent_Base.OpenAI', return_value=client), patch('Action.BiomedParsePredictor.predict', new=predict), \
                patch.dict('os.environ', {'nnUNet_raw': str(self.root)}):
            agent = MedicalSFUDAAgent('Liver', str(source), str(self.root / 'results'), 'test', 'http://unused')
            output = Path(agent.run(run_Knowledge_Distillation=False))
        plan = json.loads(next((self.root / 'results').glob('adaptation_plan_*.json')).read_text())
        self.assertEqual(len(requests), 7)  # Perception, initial Reflection, five reassessments.
        self.assertEqual(len(plan['reflection_history']), 6)
        for history in plan['reflection_history']:
            self.assertEqual(history['evaluations'][0]['vlm_evaluation']['final_decision'], 'REJECT')
            if history['iteration']:
                self.assertEqual(history['tool_results']['a']['status'], 'success')
        status = plan['reflection']['final_status']
        self.assertTrue(status['refinement_completed'])
        self.assertEqual(status['attempts'], {'a': 5})
        self.assertEqual(status['retained_cases'], [])
        self.assertEqual(status['excluded_cases'], ['a'])
        self.assertTrue(plan['current_refinement_plan']['stop_refinement'])
        self.assertEqual(plan['current_refinement_plan']['actions'], {})
        self.assertIn('budget', plan['current_refinement_plan']['reason'])
        self.assertEqual(json.loads((output / 'dataset.json').read_text())['numTraining'], 0)
        archived = load_volume(Path(status['archive']) / 'labelsTr' / 'a.nii.gz').get_fdata()
        self.assertEqual(archived[18, 18, 1], 0)  # TopologyCleaner actually changed the label.
        self.assertTrue((source / 'imagesTr' / 'a_0000.nii.gz').is_file())
        self.assertFalse((source / 'labelsTr').exists())

    def test_real_openai_sdk_http_contract_runs_complete_reflection_loop(self):
        source = self.dataset(labels=False)
        checkpoint = self.root / 'unusable_source_wiring_fixture.pth'
        checkpoint.write_bytes(b'Training must never be reached when all samples are rejected')
        packets = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                packets.append(request)
                if 'Senior Radiologist' in request['messages'][0]['content']:
                    result = {'corrected_targets': ['Liver'], 'primary_tool': 'BiomedParse', 'modality': 'CT',
                        'anatomical_site': 'Abdomen', 'image_characteristics': {'contrast_rating': 2},
                        'strategy_recommendations': {}}
                else:
                    result = critic('UNCERTAIN')
                body = json.dumps({'id': 'local-contract-check', 'object': 'chat.completion', 'created': 0,
                    'model': request['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': json.dumps(result)}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def predict(_predictor, data_path, output_dir, **kwargs):
            with np.load(Path(data_path) / 'a.npz', allow_pickle=True) as data:
                nib.save(nib.Nifti1Image(self.masks, data['affine']), Path(output_dir) / 'a.nii.gz')
            return output_dir
        try:
            with patch('Action.BiomedParsePredictor.predict', new=predict), \
                    patch.dict('os.environ', {'nnUNet_raw': str(self.root), 'NO_PROXY': '127.0.0.1,localhost'}), \
                    patch.object(FinetuningModule, 'run_preprocessing') as preprocess, \
                    patch.object(FinetuningModule, 'run_training') as train:
                agent = MedicalSFUDAAgent('Liver', str(source), str(self.root / 'http_results'),
                    'local-test', f'http://127.0.0.1:{server.server_port}/v1', source_checkpoint=str(checkpoint))
                with self.assertRaisesRegex(ValueError, 'No ACCEPT samples'):
                    agent.run()
                preprocess.assert_not_called()
                train.assert_not_called()
                plan = json.loads(next((self.root / 'http_results').glob('adaptation_plan_*.json')).read_text())
                output = Path(plan['constructed_dataset_path'])
            self.assertEqual(len(packets), 7)
            for packet in packets:
                self.assertEqual(packet['model'], 'Qwen3-VL-32B-Instruct')
                self.assertEqual(packet['temperature'], 0.1)
                self.assertEqual(packet['max_tokens'], 4096)
            self.assertEqual(packets[1]['messages'][0]['content'], PROMPT_REGISTRY['REFLECTION_CRITIC'])
            self.assertEqual(json.loads((output / 'dataset.json').read_text())['numTraining'], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_nnunet_paths_override_installed_hardcoded_defaults(self):
        from nnunet_runtime import configure_paths
        paths = SimpleNamespace(nnUNet_raw='/wrong/raw', nnUNet_preprocessed='/wrong/preprocessed', nnUNet_results='/wrong/results')
        env = {name: str(self.root / name) for name in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')}
        with patch.dict('sys.modules', {'nnunetv2': SimpleNamespace(paths=paths)}), patch.dict('os.environ', env):
            configure_paths()
        for name, value in env.items():
            self.assertEqual(getattr(paths, name), value)

    def test_preprocessing_rejects_different_source_class_ids_before_execution(self):
        import torch
        self.dataset()
        checkpoint = self.root / 'wrong_classes.pth'
        torch.save({'init_args': {'configuration': '2d', 'plans': {'configurations': {'2d': {}}},
            'dataset_json': {'channel_names': {'0': 'MR'}, 'labels': {'background': 0, 'Liver': 1, 'Kidney': 2}}}}, checkpoint)
        training = FinetuningModule(LOGGER, str(self.root))
        with patch.dict('os.environ', {'nnUNet_raw': str(self.root)}), patch.object(training, '_execute_command') as execute:
            with self.assertRaisesRegex(ValueError, 'class IDs'):
                training.run_preprocessing(123, str(checkpoint))
        execute.assert_not_called()

    def test_missing_evaluation_is_excluded_and_tool_error_finalizes(self):
        path = self.dataset(cases=('a', 'b'))
        reflection = ReflectionModule(None, str(self.root), 'model', LOGGER)
        summary = reflection._summarize_evaluations([{'case_name': 'a', 'vlm_evaluation': critic('ACCEPT')},
            {'case_name': 'b', 'vlm_evaluation': critic('REJECT')}])
        plan_path = self.root / 'failure_plan.json'
        plan_path.write_text(json.dumps({'reflection': summary, 'target_organs': ['Liver']}))
        agent = MedicalSFUDAAgent.__new__(MedicalSFUDAAgent)
        agent.reflection, agent.action = reflection, SimpleNamespace(execute_refinement_actions=lambda *args:
            (_ for _ in ()).throw(RuntimeError('action interrupted')))
        agent.logger, agent.result_dir, agent.max_refinement_iterations = LOGGER, str(self.root), 5
        with self.assertRaisesRegex(RuntimeError, 'action interrupted'):
            agent._run_refinement(str(plan_path), str(path))
        self.assertEqual(json.loads((path / 'dataset.json').read_text())['numTraining'], 1)
        self.assertFalse((path / 'labelsTr' / 'b.nii.gz').exists())
        result = json.loads(plan_path.read_text())
        self.assertFalse(result['reflection']['final_status']['refinement_completed'])
        self.assertTrue(result['current_refinement_plan']['stop_refinement'])
        self.assertIn('interrupted', result['current_refinement_plan']['reason'])

    def test_sam3_isolates_organ_sessions_and_uses_requested_crop_axis(self):
        path = self.dataset()
        output = self.root / 'sam_output'; output.mkdir()
        requests = []
        class Backend:
            def handle_request(self, req):
                requests.append(req)
                if req['type'] == 'start_session':
                    return {'session_id': str(len(requests))}
            def handle_stream_request(self, req):
                mask = np.zeros((24, 24), dtype=bool)
                mask[3:8, 3:8] = True if len([r for r in requests if r['type'] == 'start_session']) == 1 else False
                if not mask.any(): mask[14:19, 3:8] = True
                yield {'frame_index': 1, 'outputs': {'out_binary_masks': [mask]}}
        predictor = StandaloneSAM3Predictor.__new__(StandaloneSAM3Predictor)
        predictor.predictor = Backend()
        predictor.process_case('a', str(path / 'imagesTr' / 'a_0000.nii.gz'),
            [{'obj_id': 1, 'text': 'Left Kidney'}, {'obj_id': 2, 'text': 'Right Kidney'}], str(output), str(self.root / 'frames'))
        starts = [r for r in requests if r['type'] == 'start_session']
        closes = [r for r in requests if r['type'] == 'close_session']
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(closes), 2)
        mask = load_volume(output / 'a.nii.gz').get_fdata()
        self.assertEqual(set(np.unique(mask)), {0, 1, 2})
        transposed = predictor._load_and_transpose(str(path / 'imagesTr' / 'a_0000.nii.gz'), slice_axis=1)
        self.assertEqual(transposed[-1], [1, 0, 2])

    def test_tiny_roi_is_resized_without_changing_box_coordinates_or_geometry(self):
        shape = (2, 6, 4)
        image = self.root / 'tiny.nii.gz'
        nib.save(nib.Nifti1Image(np.arange(48, dtype=np.float32).reshape(shape), self.affine), image)
        output = self.root / 'tiny_output'; output.mkdir()
        requests = []
        class Backend:
            def handle_request(self, request):
                requests.append(request)
                return {'session_id': 'tiny'} if request['type'] == 'start_session' else {}
            def handle_stream_request(self, request):
                frame = Path(requests[0]['resource_path']) / '00000.jpg'
                with Image.open(frame) as img:
                    width, height = img.size
                self_size = (height, width)
                assert self_size == (64, 192)
                yield {'frame_index': 2, 'outputs': {'out_binary_masks': np.ones((1, height, width), dtype=bool)}}
        predictor = StandaloneSAM3Predictor.__new__(StandaloneSAM3Predictor)
        predictor.predictor = Backend()
        predictor.process_case('tiny', str(image), [{'obj_id': 1, 'text': 'rectangle', 'box': [1, 0, 3, 1], 'key_slice': 2}],
                               str(output), str(self.root / 'tiny_frames'), slice_axis=2)
        prompt = next(r for r in requests if r['type'] == 'add_prompt')
        np.testing.assert_allclose(prompt['bounding_boxes'], [[1/6, 0, 0.5, 0.5]])
        mask = load_volume(output / 'tiny.nii.gz')
        self.assertEqual(mask.shape, shape)
        np.testing.assert_allclose(mask.affine, self.affine)
        self.assertTrue(np.all(mask.get_fdata()[..., 2] == 1))


if __name__ == '__main__':
    unittest.main()
