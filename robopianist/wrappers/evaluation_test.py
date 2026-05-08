# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for evaluation.py."""

from types import SimpleNamespace

import dm_env
import numpy as np
from absl.testing import absltest
from dm_env import specs

from robopianist.models.piano.midi_module import MAX_KEY_VEL as _MAX_KEY_VEL, QVEL_MIN as _QVEL_MIN
from robopianist.suite.tasks.piano_with_shadow_hands import ScoreKeyMetadata
from robopianist.wrappers.evaluation import MidiEvaluationWrapper


class _FakePiano:
    def __init__(self, key_id: int, onset_qvel: float) -> None:
        self.n_keys = 88
        self.activation = np.zeros((self.n_keys,), dtype=bool)
        self.sustain_activation = np.zeros((1,), dtype=bool)
        self._onset_velocities = np.zeros((self.n_keys,), dtype=np.float64)
        self._key_id = key_id
        self._onset_qvel = onset_qvel

    def set_new_onset(self) -> None:
        self.activation[:] = False
        self.activation[self._key_id] = True
        self._onset_velocities[:] = 0.0
        self._onset_velocities[self._key_id] = self._onset_qvel


class _FakeTask:
    def __init__(
        self,
        key_id: int,
        metadata: ScoreKeyMetadata,
        score_velocity: int,
        onset_qvel: float,
    ) -> None:
        self._prev_activation = np.zeros((88,), dtype=bool)
        self._t_idx = 1
        self._notes = [[SimpleNamespace(key=key_id, velocity=score_velocity)]]
        self._sustains = [int(metadata.score_sustain)]
        self.reward_fn = SimpleNamespace(reward_terms={})
        self.piano = _FakePiano(key_id, onset_qvel)
        self._metadata = metadata

    def get_score_key_metadata(self, t_idx: int, key_id: int) -> ScoreKeyMetadata:
        del t_idx, key_id  # Unused in the fixed test setup.
        return self._metadata


class _FakeEnv(dm_env.Environment):
    def __init__(self, task: _FakeTask) -> None:
        self.task = task

    def reset(self) -> dm_env.TimeStep:
        self.task._prev_activation[:] = False
        self.task.piano.activation[:] = False
        return dm_env.restart(observation={})

    def step(self, action) -> dm_env.TimeStep:
        del action  # Unused.
        self.task.piano.set_new_onset()
        return dm_env.termination(reward=0.0, observation={})

    def observation_spec(self):
        return {}

    def action_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name="action")

    def discount_spec(self):
        return specs.Array(shape=(), dtype=np.float32, name="discount")

    def reward_spec(self):
        return specs.Array(shape=(), dtype=np.float32, name="reward")


class MidiEvaluationWrapperTest(absltest.TestCase):
    def test_trace_marks_active_but_not_true_onset_as_unmatched(self) -> None:
        key_id = 40
        onset_qvel = 2.0
        metadata = ScoreKeyMetadata(
            score_key_active=True,
            gt_is_true_onset=False,
            score_sustain=True,
            gt_active_midi_vel=80,
            gt_true_onset_midi_vel=None,
        )
        env = MidiEvaluationWrapper(
            _FakeEnv(
                _FakeTask(
                    key_id,
                    metadata,
                    score_velocity=80,
                    onset_qvel=onset_qvel,
                )
            )
        )

        env.reset()
        env.step(np.zeros((1,), dtype=np.float32))

        trace = env.get_episode_velocity_trace()
        self.assertLen(trace, 1)
        row = trace[0]
        self.assertFalse(row["matched"])
        self.assertEqual(row["gt_midi_vel"], -1)
        self.assertEqual(row["score_active_midi_vel"], 80)
        self.assertTrue(row["score_key_active"])
        self.assertFalse(row["gt_is_true_onset"])
        self.assertTrue(row["score_sustain"])
        self.assertTrue(row["robot_new_onset"])
        metrics = env.get_velocity_metrics()
        self.assertEmpty(metrics)

    def test_trace_uses_true_onset_velocity_when_present(self) -> None:
        key_id = 40
        onset_qvel = 2.0
        gt_midi_vel = 64
        metadata = ScoreKeyMetadata(
            score_key_active=True,
            gt_is_true_onset=True,
            score_sustain=False,
            gt_active_midi_vel=gt_midi_vel,
            gt_true_onset_midi_vel=gt_midi_vel,
        )
        env = MidiEvaluationWrapper(
            _FakeEnv(
                _FakeTask(
                    key_id,
                    metadata,
                    score_velocity=gt_midi_vel,
                    onset_qvel=onset_qvel,
                )
            )
        )

        env.reset()
        env.step(np.zeros((1,), dtype=np.float32))

        trace = env.get_episode_velocity_trace()
        self.assertLen(trace, 1)
        row = trace[0]
        expected_robot_midi_vel = int(
            np.clip((onset_qvel - _QVEL_MIN) / (_MAX_KEY_VEL - _QVEL_MIN) * 126, 0, 126)
        ) + 1
        expected_needed_qvel = (gt_midi_vel - 1) / 126.0 * (_MAX_KEY_VEL - _QVEL_MIN) + _QVEL_MIN

        self.assertTrue(row["matched"])
        self.assertEqual(row["gt_midi_vel"], gt_midi_vel)
        self.assertEqual(row["score_active_midi_vel"], gt_midi_vel)
        self.assertTrue(row["score_key_active"])
        self.assertTrue(row["gt_is_true_onset"])
        self.assertFalse(row["score_sustain"])
        self.assertTrue(row["robot_new_onset"])
        self.assertEqual(row["robot_midi_vel"], expected_robot_midi_vel)
        self.assertAlmostEqual(row["needed_qvel"], round(expected_needed_qvel, 4))
        metrics = env.get_velocity_metrics()
        expected_error = abs(expected_robot_midi_vel - gt_midi_vel)
        self.assertAlmostEqual(metrics["velocity_mae"], float(expected_error))
        self.assertAlmostEqual(metrics["velocity_bias"], float(expected_robot_midi_vel - gt_midi_vel))
        self.assertIn("perceptual_dynamics_score", metrics)
        # With a single matched onset, loud_corr is nan → s_corr=0 → PDS=0.
        self.assertEqual(metrics["perceptual_dynamics_score"], 0.0)

    def test_pds_is_present_and_bounded(self) -> None:
        """PDS is present in velocity metrics and lies in [0, 1]."""
        key_id = 40
        onset_qvel = 2.0
        gt_midi_vel = 64
        metadata = ScoreKeyMetadata(
            score_key_active=True,
            gt_is_true_onset=True,
            score_sustain=False,
            gt_active_midi_vel=gt_midi_vel,
            gt_true_onset_midi_vel=gt_midi_vel,
        )
        env = MidiEvaluationWrapper(
            _FakeEnv(
                _FakeTask(key_id, metadata, score_velocity=gt_midi_vel, onset_qvel=onset_qvel)
            )
        )
        env.reset()
        env.step(np.zeros((1,), dtype=np.float32))

        metrics = env.get_velocity_metrics()
        self.assertIn("perceptual_dynamics_score", metrics)
        pds = metrics["perceptual_dynamics_score"]
        # Single onset → loud_corr is nan → s_corr=0 → PDS=0.
        self.assertEqual(pds, 0.0)


if __name__ == "__main__":
    absltest.main()
