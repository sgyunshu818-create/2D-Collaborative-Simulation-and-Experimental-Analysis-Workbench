"""Integration checks for selectable units, evidence reading and the desktop flow."""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from sim_app.app import App
from sim_app.experiment_results import ResultLoadError, _manifest_directory, load_results
from sim_app.recording import export_run
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation
from sim_app.tk_runtime import create_root
from sim_app.workbench import Workbench


PROJECT = Path(__file__).resolve().parents[1]


def basic():
    return json.loads((PROJECT / 'configs' / 'basic_scene.json').read_text(encoding='utf-8'))


class InspectorTests(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        payload = basic()
        units = []
        for i in range(3):
            for spec in payload['units']:
                spec = copy.deepcopy(spec)
                spec['id'] += f'_{i}'
                units.append(spec)
        payload['units'] = units
        self.sim = Simulation(scene_from_data(payload, Path('twelve.json')))
        self.renderer = Renderer(pygame.Surface(WINDOW_SIZE))

    def test_last_of_twelve_overlapping_units_selectable_from_list(self):
        self.renderer.unit_offset = 7
        self.renderer.draw(self.sim)
        rect, unit_id = self.renderer.unit_rows[-1]
        self.assertEqual(unit_id, self.sim.units[-1].id)
        self.assertTrue(self.renderer.select_at(rect.center, self.sim))
        self.assertEqual(self.renderer.selected_unit_id, unit_id)
        self.assertEqual(len(self.renderer.visible_units(self.sim)), 12)

    def test_search_accepts_keydown_and_textinput_without_double_text(self):
        self.renderer.search_active = True
        app = App(self.sim, self.renderer)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, unicode='b'))
        app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text='b'))
        app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text='lue'))
        self.assertEqual(self.renderer.unit_search, 'blue')
        self.assertEqual(len(self.renderer.visible_units(self.sim)), 6)
        self.assertEqual(self.sim.step_count, 0)


class EvidenceAdapterTests(unittest.TestCase):
    def test_relocated_repeat_prefers_matching_local_dataset_over_existing_old_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original' / 'artifacts' / 'marker_experiment'
            local = root / 'relocated' / 'artifacts' / 'marker_experiment'
            repeat = local.parent / 'marker_repeat5' / 'batch'
            repeat.mkdir(parents=True)
            for candidate in (original, local):
                candidate.mkdir(parents=True)
                (candidate / 'manifest.csv').write_text('image_id,path\n', encoding='utf-8')
                (candidate / 'manifest.json').write_text('{"dataset":"same"}', encoding='utf-8')
            protocol = {'source': str(original), 'manifest_sha256':
                        hashlib.sha256((original / 'manifest.json').read_bytes()).hexdigest()}
            self.assertEqual(_manifest_directory(repeat, protocol), local)

    def test_unrelated_adjacent_dataset_does_not_replace_recorded_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original' / 'marker_experiment'
            local = root / 'relocated' / 'artifacts' / 'marker_experiment'
            repeat = local.parent / 'marker_repeat5' / 'batch'
            repeat.mkdir(parents=True)
            for candidate, identity in ((original, 'correct'), (local, 'unrelated')):
                candidate.mkdir(parents=True)
                (candidate / 'manifest.csv').write_text('image_id,path\n', encoding='utf-8')
                (candidate / 'manifest.json').write_text(identity, encoding='utf-8')
            protocol = {'source': str(original), 'manifest_sha256':
                        hashlib.sha256((original / 'manifest.json').read_bytes()).hexdigest()}
            self.assertEqual(_manifest_directory(repeat, protocol), original)
            (original / 'manifest.json').write_text('also changed', encoding='utf-8')
            self.assertIsNone(_manifest_directory(repeat, protocol))

    def test_repeat_calls_do_not_inflate_independent_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            columns = ['image_id','original_id','condition','label','prediction','algorithm','elapsed_ns','round','correct']
            with (root/'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for index in (1,2):
                    writer.writerow(dict(image_id='a-noise',original_id='a',condition='noise',label='triangle',prediction='square',algorithm='baseline',elapsed_ns=2000,round=index,correct=1))
            result = load_results(root)
            self.assertEqual((result['independent_originals'], result['round_count']), (1,2))
            self.assertEqual(len(result['failures']), 2)
            self.assertEqual(result['metrics'][0]['call_count'], 2)
            self.assertEqual(result['metrics'][0]['confusion']['triangle']['square'], 2)

    def test_bad_predictions_report_error_without_writing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/'predictions.csv'
            file.write_text('image_id\na\n',encoding='utf-8')
            before = file.read_bytes()
            with self.assertRaises(ResultLoadError):
                load_results(folder)
            self.assertEqual(file.read_bytes(), before)


class WorkbenchFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.root = create_root()
        self.root.withdraw()
        self.app = Workbench(self.root, self.folder/'runs')
        self.addCleanup(self.clean)
        error = patch.object(self.app, '_error', side_effect=lambda exc: (_ for _ in ()).throw(exc))
        error.start()
        self.addCleanup(error.stop)
        self.drain()

    def clean(self):
        self.app.closed = True
        self.root.after_cancel(self.app._after_id)
        self.app.pool.shutdown(wait=True)
        self.root.destroy()

    def drain(self):
        deadline = time.monotonic()+15
        while self.app.jobs and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.02)
        self.root.update()
        self.assertFalse(self.app.jobs, 'Background readers did not complete')

    def save_run(self, name, speed):
        payload = basic()
        payload['units'][0]['speed'] = speed
        sim = Simulation(scene_from_data(payload, Path('input.json')))
        sim.start()
        sim.advance(1)
        sim.pause()
        directory = self.folder/'runs'/name
        export_run(sim, directory)
        return str((directory/'run.json').resolve())

    def test_history_compare_export_copy_and_restart_are_connected(self):
        left = self.save_run('first',45)
        right = self.save_run('second',30)
        self.app.refresh_history()
        self.drain()
        self.assertEqual(len(self.app.history_entries),2)
        self.app.history_tree.selection_set(left)
        self.root.update()
        self.drain()
        self.assertEqual(self.app.current_payload['scene']['units'][0]['speed'],45)
        self.assertIn('任务未完成',self.app.result_text.get('1.0','end'))
        self.app.history_tree.selection_set((left,right))
        self.app.compare_selected()
        self.drain()
        self.assertIn('speed', self.app.compare_text.get('1.0','end'))
        output = self.folder/'comparison.csv'
        with patch('sim_app.workbench.filedialog.asksaveasfilename',return_value=str(output)):
            self.app.export_selected()
        self.drain()
        self.assertIn(left,output.read_text(encoding='utf-8-sig'))
        self.app.history_tree.selection_set(left)
        self.root.update()
        self.drain()
        self.app.assign_compare('left')
        self.app.history_tree.selection_set(right)
        self.root.update()
        self.drain()
        self.app.assign_compare('right')
        self.assertEqual(len(self.app.history_tree.selection()), 1)
        self.app.compare_selected()
        self.drain()
        self.assertIn('speed', self.app.compare_text.get('1.0','end'))
        self.app.history_tree.selection_set(right)
        self.root.update()
        self.drain()
        self.app.copy_input()
        self.drain()
        self.assertEqual(self.app.editor.get_document().data['units'][0]['speed'],30)
        self.assertEqual(self.app.editor.get_document().source_path,Path(right))
        self.app.refresh_history()
        self.drain()
        self.assertEqual(len(self.app.history_tree.get_children()),2)

    def test_launched_input_is_independent_of_subsequent_saved_edits(self):
        source = self.folder/'scene.json'
        payload=basic()
        source.write_text(json.dumps(payload),encoding='utf-8')
        calls=[]
        with patch('sim_app.workbench.PROJECT_ROOT',self.folder), patch.object(self.app,'_launch',side_effect=lambda args,label: calls.append(args)):
            self.app.launch_scene(source)
        frozen=Path(calls[0][1])
        payload['units'][0]['speed']=10
        source.write_text(json.dumps(payload),encoding='utf-8')
        self.assertEqual(json.loads(frozen.read_text(encoding='utf-8'))['units'][0]['speed'],45)


if __name__ == '__main__':
    unittest.main()
