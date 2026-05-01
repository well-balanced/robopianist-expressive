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

"""Tests for piano_with_shadow_hands_test.py."""

import itertools
from typing import Optional, Sequence

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer
from mujoco_utils import spec_utils
from note_seq.protobuf import music_pb2

from robopianist.models.piano.midi_module import QVEL_MIN as _QVEL_MIN
from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands

_HAS_LEGACY_MATCHED_ONSET_API = all(
    hasattr(piano_with_shadow_hands.PianoWithShadowHands, name)
    for name in ("match_score_onset", "_compute_key_press_reward_v2")
)


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 for 2 dts.
    seq.notes.add(
        start_time=0.0,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,  # Right hand index.
    )
    # G5 for 1 dt.
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=0,  # Left hand thumb.
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_test_midi_with_sustain(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    seq.notes.add(
        start_time=0.0,
        end_time=1 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,
    )
    seq.control_changes.add(
        time=0.0,
        control_number=64,
        control_value=64,
        instrument=0,
    )
    seq.control_changes.add(
        time=3 * dt,
        control_number=64,
        control_value=0,
        instrument=0,
    )

    seq.total_time = 4 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_test_midi_repeated_same_key(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 onset at t=0 for 1 dt.
    seq.notes.add(
        start_time=0.0,
        end_time=1 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,
    )
    # Same key again at t=4 for 1 dt.
    seq.notes.add(
        start_time=4 * dt,
        end_time=5 * dt,
        velocity=72,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,
    )

    seq.total_time = 5 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_test_midi_intervening_other_key(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    seq.notes.add(
        start_time=0.0,
        end_time=1 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,
    )
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=76,
        pitch=midi_file.note_name_to_midi_number("D6"),
        part=2,
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_env(
    control_timestep: float = 0.01,
    n_steps_lookahead: int = 0,
    n_seconds_lookahead: Optional[float] = None,
    wrong_press_termination: bool = False,
    disable_fingering_reward: bool = False,
    midi: Optional[midi_file.MidiFile] = None,
    style_velocity_scale: float = 1.0,
    velocity_onset_window_steps: int = 0,
    style_velocity_scale_choices: Optional[Sequence[float]] = None,
    onset_accuracy_reward_coef: float = 0.0,
    n_steps_velocity_lookahead: int = 3,
) -> composer.Environment:
    task_kwargs = dict(
        midi=midi or _get_test_midi(dt=control_timestep),
        n_steps_lookahead=n_steps_lookahead,
        n_seconds_lookahead=n_seconds_lookahead,
        control_timestep=control_timestep,
        wrong_press_termination=wrong_press_termination,
        change_color_on_activation=True,
        disable_fingering_reward=disable_fingering_reward,
        style_velocity_scale=style_velocity_scale,
        style_velocity_scale_choices=style_velocity_scale_choices,
        onset_accuracy_reward_coef=onset_accuracy_reward_coef,
        n_steps_velocity_lookahead=n_steps_velocity_lookahead,
    )
    if velocity_onset_window_steps != 0:
        task_kwargs["velocity_onset_window_steps"] = velocity_onset_window_steps
    task = piano_with_shadow_hands.PianoWithShadowHands(**task_kwargs)
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class PianoWithShadowHandsTest(parameterized.TestCase):
    @parameterized.parameters(True, False)
    def test_observables(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        timestep = env.reset()

        # Piano observables.
        self.assertIn("piano/state", timestep.observation)
        self.assertIn("piano/sustain_state", timestep.observation)

        # Goal observables.
        self.assertIn("goal", timestep.observation)
        self.assertIn("goal_velocity", timestep.observation)
        self.assertIn("goal_true_onset", timestep.observation)
        self.assertIn("goal_true_onset_velocity", timestep.observation)
        if disable_fingering_reward:
            self.assertNotIn("fingering", timestep.observation)
        else:
            self.assertIn("fingering", timestep.observation)

        # Hand observables.
        for name in ["rh_shadow_hand", "lh_shadow_hand"]:
            self.assertIn(f"{name}/joints_pos", timestep.observation)
            # self.assertIn(f"{name}/position", timestep.observation)

    def test_action_spec(self) -> None:
        env = _get_env()
        rh_action_spec = env.task.right_hand.action_spec(env.physics)
        lh_action_spec = env.task.left_hand.action_spec(env.physics)
        combined_spec = spec_utils.merge_specs([rh_action_spec, lh_action_spec])
        actual_shape = env.action_spec().shape[0] - 1  # Don't include sustain pedal.
        expected_shape = combined_spec.shape[0]
        self.assertEqual(actual_shape, expected_shape)

        right_action = np.random.uniform(
            low=rh_action_spec.minimum, high=rh_action_spec.maximum
        ).astype(rh_action_spec.dtype)
        left_action = np.random.uniform(
            low=lh_action_spec.minimum, high=lh_action_spec.maximum
        ).astype(lh_action_spec.dtype)
        action = np.concatenate([right_action, left_action, [0]])
        env.task.before_step(env.physics, action, env.random_state)

        actual_rh_action = env.physics.bind(env.task.right_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_rh_action, right_action)
        actual_lh_action = env.physics.bind(env.task.left_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_lh_action, left_action)

    def test_termination_and_discount(self) -> None:
        env = _get_env()
        action_spec = env.action_spec()
        env.reset()

        # With a dt of 0.01 and a 3 dt long midi, the episode should end after 4 steps.
        zero_action = np.zeros(action_spec.shape)
        for _ in range(3):
            timestep = env.step(zero_action)
            self.assertFalse(env.task.should_terminate_episode(env.physics))
            np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

        # 1 more step to terminate.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # No failure, so discount should be 1.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

    @parameterized.parameters(itertools.product([0.01, 0.05, 0.1], [0, 0.01, 0.1, 1]))
    def test_n_seconds_lookahead(
        self, control_timestep: float, n_seconds_lookahead: float
    ) -> None:
        env = _get_env(
            control_timestep=control_timestep, n_seconds_lookahead=n_seconds_lookahead
        )

        actual_n_steps_lookahead = env.task._n_steps_lookahead
        expected_n_steps_lookahead = int(
            np.ceil(n_seconds_lookahead / control_timestep)
        )
        self.assertEqual(actual_n_steps_lookahead, expected_n_steps_lookahead)

    @parameterized.parameters(0, 1, 2, 5)
    def test_goal_observable_lookahead(self, n_steps_lookahead: int) -> None:
        env = _get_env(control_timestep=0.01, n_steps_lookahead=n_steps_lookahead)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        sustains = note_traj.sustains
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_goal = np.zeros((n_steps_lookahead + 1, env.task.piano.n_keys + 1))

            t_start = i
            t_end = min(i + n_steps_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                for note in notes[t]:
                    expected_goal[j, note.key] = 1.0
                expected_goal[j, -1] = sustains[t]

            actual_goal = timestep.observation["goal"]
            np.testing.assert_array_equal(actual_goal, expected_goal.ravel())

            # Check that the 0th goal is always the goal at the current timestep.
            expected_reward_goal = np.zeros((env.task.piano.n_keys + 1,))
            for note in notes[i]:
                expected_reward_goal[note.key] = 1.0
            expected_reward_goal[-1] = sustains[i]
            actual_current = timestep.observation["goal"][0 : env.task.piano.n_keys + 1]
            np.testing.assert_array_equal(actual_current, expected_reward_goal)

            timestep = env.step(zero_action)

            # The observable and reward cache now both keep the binary press target.
            np.testing.assert_array_equal(expected_reward_goal, env.task._goal_current)

    @parameterized.parameters(0, 1, 3)
    def test_goal_velocity_observable_lookahead(self, n_steps_velocity_lookahead: int) -> None:
        env = _get_env(
            control_timestep=0.01,
            midi=_get_test_midi(dt=0.01),
            n_steps_velocity_lookahead=n_steps_velocity_lookahead,
        )
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes

        for i in range(len(notes)):
            expected_goal_velocity = np.zeros(
                (n_steps_velocity_lookahead + 1, env.task.piano.n_keys)
            )
            t_start = i
            t_end = min(i + n_steps_velocity_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                for note in notes[t]:
                    expected_goal_velocity[j, note.key] = note.velocity / 127.0

            actual_goal_velocity = timestep.observation["goal_velocity"]
            np.testing.assert_array_equal(
                actual_goal_velocity, expected_goal_velocity.ravel()
            )
            timestep = env.step(zero_action)

    @parameterized.parameters(0, 1, 3)
    def test_goal_true_onset_observable_lookahead(
        self, n_steps_velocity_lookahead: int
    ) -> None:
        env = _get_env(
            control_timestep=0.01,
            midi=_get_test_midi(dt=0.01),
            n_steps_velocity_lookahead=n_steps_velocity_lookahead,
        )
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        prev_active_keys = set()

        for i in range(len(notes)):
            expected_goal_true_onset = np.zeros(
                (n_steps_velocity_lookahead + 1, env.task.piano.n_keys)
            )
            local_prev_active_keys = prev_active_keys.copy()
            t_start = i
            t_end = min(i + n_steps_velocity_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                active_keys = {note.key for note in notes[t]}
                true_onset_keys = active_keys - local_prev_active_keys
                for key in true_onset_keys:
                    expected_goal_true_onset[j, key] = 1.0
                local_prev_active_keys = active_keys

            actual_goal_true_onset = timestep.observation["goal_true_onset"]
            np.testing.assert_array_equal(
                actual_goal_true_onset, expected_goal_true_onset.ravel()
            )

            prev_active_keys = {note.key for note in notes[i]}
            timestep = env.step(zero_action)

    @parameterized.parameters(0, 1, 3)
    def test_goal_true_onset_velocity_observable_lookahead(
        self, n_steps_velocity_lookahead: int
    ) -> None:
        env = _get_env(
            control_timestep=0.01,
            midi=_get_test_midi(dt=0.01),
            n_steps_velocity_lookahead=n_steps_velocity_lookahead,
        )
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        prev_active_keys = set()

        for i in range(len(notes)):
            expected_goal_true_onset_velocity = np.zeros(
                (n_steps_velocity_lookahead + 1, env.task.piano.n_keys)
            )
            local_prev_active_keys = prev_active_keys.copy()
            t_start = i
            t_end = min(i + n_steps_velocity_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                active_notes = {note.key: note.velocity for note in notes[t]}
                true_onset_keys = set(active_notes) - local_prev_active_keys
                for key in true_onset_keys:
                    expected_goal_true_onset_velocity[j, key] = (
                        active_notes[key] / 127.0
                    )
                local_prev_active_keys = set(active_notes)

            actual_goal_true_onset_velocity = timestep.observation[
                "goal_true_onset_velocity"
            ]
            np.testing.assert_array_equal(
                actual_goal_true_onset_velocity,
                expected_goal_true_onset_velocity.ravel(),
            )

            prev_active_keys = {note.key for note in notes[i]}
            timestep = env.step(zero_action)

    def test_fingering_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_fingering = np.zeros((2, 5))
            idxs = [note.fingering for note in notes[i]]
            rh_idxs = [idx for idx in idxs if idx < 5]
            lh_idxs = [idx - 5 for idx in idxs if idx >= 5]
            expected_fingering[0, rh_idxs] = 1.0
            expected_fingering[1, lh_idxs] = 1.0

            actual_fingering = timestep.observation["fingering"]
            np.testing.assert_array_equal(actual_fingering, expected_fingering.ravel())

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the fingering information for the
            # current timestep to compute the reward. Let's check that it matches the
            # expected one.
            actual_rh_current = [r[1] for r in env.task._rh_keys_current]
            np.testing.assert_array_equal(rh_idxs, actual_rh_current)
            actual_lh_current = [r[1] for r in env.task._lh_keys_current]
            np.testing.assert_array_equal(lh_idxs, actual_lh_current)

    def test_mixed_style_velocity_scale_changes_goal_targets(self) -> None:
        env = _get_env(
            control_timestep=0.01,
            midi=_get_test_midi(dt=0.01),
            style_velocity_scale_choices=(0.5, 1.5),
        )
        key_id = midi_file.note_name_to_key_number("C6")

        seen_scales = set()
        for _ in range(12):
            timestep = env.reset()
            scale = env.task.current_style_velocity_scale
            seen_scales.add(scale)
            self.assertIn(scale, (0.5, 1.5))
            expected_velocity = round(80 * scale) / 127.0
            self.assertEqual(timestep.observation["goal"][key_id], 1.0)
            self.assertAlmostEqual(
                timestep.observation["goal_velocity"][key_id], expected_velocity
            )

        self.assertEqual(seen_scales, {0.5, 1.5})

    def test_fixed_style_velocity_scale_changes_goal_targets(self) -> None:
        env = _get_env(
            control_timestep=0.01,
            midi=_get_test_midi(dt=0.01),
            style_velocity_scale=0.75,
        )
        key_id = midi_file.note_name_to_key_number("C6")

        timestep = env.reset()
        self.assertAlmostEqual(env.task.current_style_velocity_scale, 0.75)
        expected_velocity = round(80 * 0.75) / 127.0
        self.assertEqual(timestep.observation["goal"][key_id], 1.0)
        self.assertAlmostEqual(
            timestep.observation["goal_velocity"][key_id], expected_velocity
        )

    def test_style_velocity_scale_choices_are_exposed(self) -> None:
        env = _get_env(style_velocity_scale_choices=(0.8, 1.0, 1.2))
        self.assertEqual(env.task.style_velocity_scale_choices, (0.8, 1.0, 1.2))

    def test_score_key_metadata_distinguishes_true_onset_from_active_note(self) -> None:
        env = _get_env(control_timestep=0.01)
        key_id = midi_file.note_name_to_key_number("C6")

        onset_metadata = env.task.get_score_key_metadata(0, key_id)
        self.assertTrue(onset_metadata.score_key_active)
        self.assertTrue(onset_metadata.gt_is_true_onset)
        self.assertFalse(onset_metadata.score_sustain)
        self.assertEqual(onset_metadata.gt_active_midi_vel, 80)
        self.assertEqual(onset_metadata.gt_true_onset_midi_vel, 80)

        held_metadata = env.task.get_score_key_metadata(1, key_id)
        self.assertTrue(held_metadata.score_key_active)
        self.assertFalse(held_metadata.gt_is_true_onset)
        self.assertFalse(held_metadata.score_sustain)
        self.assertEqual(held_metadata.gt_active_midi_vel, 80)
        self.assertIsNone(held_metadata.gt_true_onset_midi_vel)

    def test_score_key_metadata_reports_score_sustain(self) -> None:
        env = _get_env(
            control_timestep=0.01,
            disable_fingering_reward=True,
            midi=_get_test_midi_with_sustain(dt=0.01),
        )
        key_id = midi_file.note_name_to_key_number("C6")

        metadata = env.task.get_score_key_metadata(1, key_id)
        self.assertTrue(metadata.score_sustain)

    def test_onset_accuracy_reward_is_registered_only_when_enabled(self) -> None:
        disabled_env = _get_env(onset_accuracy_reward_coef=0.0)
        self.assertNotIn(
            "onset_accuracy_reward", disabled_env.task.reward_fn.reward_fns
        )

        enabled_env = _get_env(onset_accuracy_reward_coef=1.0)
        self.assertIn("onset_accuracy_reward", enabled_env.task.reward_fn.reward_fns)

    def test_onset_accuracy_reward_hits_and_misses_true_onsets(self) -> None:
        env = _get_env(onset_accuracy_reward_coef=1.0)
        task = env.task
        c6 = midi_file.note_name_to_key_number("C6")
        d6 = midi_file.note_name_to_key_number("D6")

        task._score_active_velocity_maps = [{c6: 80, d6: 72}]
        task._score_true_onset_velocity_maps = [{c6: 80, d6: 72}]
        task._t_idx = 1
        task._prev_activation[:] = False
        task.piano._activation[:] = False
        task.piano._activation[c6] = True

        reward = task._compute_onset_accuracy_reward(env.physics)
        self.assertAlmostEqual(reward, 0.1)

    def test_onset_accuracy_reward_penalizes_offscore_fp(self) -> None:
        env = _get_env(onset_accuracy_reward_coef=1.0)
        task = env.task
        c6 = midi_file.note_name_to_key_number("C6")

        task._score_active_velocity_maps = [{}]
        task._score_true_onset_velocity_maps = [{}]
        task._t_idx = 1
        task._prev_activation[:] = False
        task.piano._activation[:] = False
        task.piano._activation[c6] = True

        reward = task._compute_onset_accuracy_reward(env.physics)
        self.assertAlmostEqual(reward, -0.05)

    def test_onset_accuracy_reward_penalizes_hold_rehit(self) -> None:
        env = _get_env(onset_accuracy_reward_coef=1.0)
        task = env.task
        c6 = midi_file.note_name_to_key_number("C6")

        task._score_active_velocity_maps = [{}, {}, {c6: 80}]
        task._score_true_onset_velocity_maps = [{}, {}, {}]
        task._t_idx = 3
        task._prev_activation[:] = False
        task.piano._activation[:] = False
        task.piano._activation[c6] = True

        reward = task._compute_onset_accuracy_reward(env.physics)
        self.assertAlmostEqual(reward, -0.2)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_match_score_onset_uses_symmetric_timestep_window(self) -> None:
        window = 2
        env = _get_env(control_timestep=0.01, velocity_onset_window_steps=window)
        key_id = midi_file.note_name_to_key_number("C6")

        exact_match = env.task.match_score_onset(0, key_id)
        self.assertEqual(exact_match.matched_t_idx, 0)
        self.assertEqual(exact_match.gt_true_onset_midi_vel, 80)
        self.assertEqual(exact_match.timing_offset, 0)
        self.assertAlmostEqual(exact_match.timing_weight, 1.0)

        delayed_match = env.task.match_score_onset(1, key_id)
        self.assertEqual(delayed_match.matched_t_idx, 0)
        self.assertEqual(delayed_match.gt_true_onset_midi_vel, 80)
        self.assertEqual(delayed_match.timing_offset, 1)
        self.assertLess(delayed_match.timing_weight, 1.0)

        missed_match = env.task.match_score_onset(window + 1, key_id)
        self.assertIsNone(missed_match.matched_t_idx)
        self.assertIsNone(missed_match.gt_true_onset_midi_vel)
        self.assertEqual(missed_match.timing_weight, 0.0)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_match_score_onset_does_not_reuse_consumed_gt_onset(self) -> None:
        env = _get_env(control_timestep=0.01, velocity_onset_window_steps=2)
        key_id = midi_file.note_name_to_key_number("C6")

        first_match = env.task.match_score_onset(1, key_id, used_t_idxs=set())
        self.assertEqual(first_match.matched_t_idx, 0)
        self.assertEqual(first_match.gt_true_onset_midi_vel, 80)

        consumed_match = env.task.match_score_onset(1, key_id, used_t_idxs={0})
        self.assertIsNone(consumed_match.matched_t_idx)
        self.assertIsNone(consumed_match.gt_true_onset_midi_vel)
        self.assertEqual(consumed_match.timing_weight, 0.0)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_match_score_onset_advances_to_next_same_key_onset_after_consumption(self) -> None:
        env = _get_env(
            control_timestep=0.01,
            disable_fingering_reward=True,
            midi=_get_test_midi_repeated_same_key(dt=0.01),
            velocity_onset_window_steps=2,
        )
        key_id = midi_file.note_name_to_key_number("C6")

        first_match = env.task.match_score_onset(2, key_id, used_t_idxs=set())
        self.assertEqual(first_match.matched_t_idx, 0)
        self.assertEqual(first_match.gt_true_onset_midi_vel, 80)

        second_match = env.task.match_score_onset(2, key_id, used_t_idxs={0})
        self.assertEqual(second_match.matched_t_idx, 4)
        self.assertEqual(second_match.gt_true_onset_midi_vel, 72)
        self.assertEqual(second_match.timing_offset, -2)
        self.assertGreater(second_match.timing_weight, 0.0)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_match_score_onset_blocks_late_match_when_other_key_onset_intervenes(self) -> None:
        env = _get_env(
            control_timestep=0.01,
            disable_fingering_reward=True,
            midi=_get_test_midi_intervening_other_key(dt=0.01),
            velocity_onset_window_steps=2,
        )
        c6_key_id = midi_file.note_name_to_key_number("C6")

        blocked_match = env.task.match_score_onset(2, c6_key_id, used_t_idxs=set())
        self.assertIsNone(blocked_match.matched_t_idx)
        self.assertIsNone(blocked_match.gt_true_onset_midi_vel)
        self.assertEqual(blocked_match.timing_weight, 0.0)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_key_press_reward_v2_matches_nearby_true_onset_within_window(self) -> None:
        env = _get_env(control_timestep=0.01, velocity_onset_window_steps=2)
        env.reset()
        task = env.task
        key_id = midi_file.note_name_to_key_number("C6")

        task._goal_current = np.zeros((task.piano.n_keys + 1,), dtype=np.float64)
        task._goal_current[key_id] = 1.0
        task._prev_activation[:] = False
        task.piano._activation[:] = False
        task.piano._activation[key_id] = True
        task.piano._state[:] = 0.0
        task.piano._state[key_id] = task.piano._qpos_range[key_id, 1]
        task.piano._onset_velocities[:] = 0.0
        task.piano._onset_velocities[key_id] = 2.0

        task._t_idx = 2
        task._compute_key_press_reward_v2(env.physics)
        self.assertEqual(task._key_onset_gt_vel[key_id], 80)
        self.assertLess(task._key_onset_timing_weight[key_id], 1.0)

        task._key_onset_gt_vel[key_id] = -999.0
        task._key_onset_timing_weight[key_id] = -999.0
        task._t_idx = task._velocity_onset_window_steps + 4
        task._compute_key_press_reward_v2(env.physics)
        self.assertEqual(task._key_onset_gt_vel[key_id], -1.0)
        self.assertEqual(task._key_onset_timing_weight[key_id], 0.0)

    @absltest.skipUnless(
        _HAS_LEGACY_MATCHED_ONSET_API,
        "Legacy matched-onset helper API is not present in the current task implementation.",
    )
    def test_key_press_reward_v2_does_not_carry_velocity_penalty_into_hold(self) -> None:
        env = _get_env(control_timestep=0.01)
        env.reset()
        task = env.task
        key_id = midi_file.note_name_to_key_number("C6")

        task._goal_current = np.zeros((task.piano.n_keys + 1,), dtype=np.float64)
        task._goal_current[key_id] = 1.0
        task.piano._state[:] = 0.0
        task.piano._state[key_id] = task.piano._qpos_range[key_id, 1]
        task.piano._activation[:] = False
        task.piano._activation[key_id] = True
        task.piano._onset_velocities[:] = 0.0

        # True onset with extremely soft strike should reduce only the onset-step reward.
        task._prev_activation[:] = False
        task._t_idx = 1
        task.piano._onset_velocities[key_id] = _QVEL_MIN
        onset_rew = task._compute_key_press_reward_v2(env.physics)

        # Holding the same key on the next step should recover to the pure key-press reward.
        task._prev_activation[:] = task.piano._activation.copy()
        task._t_idx = 2
        hold_rew = task._compute_key_press_reward_v2(env.physics)

        self.assertLess(onset_rew, 1.0)
        self.assertAlmostEqual(hold_rew, 1.0)

    def test_failure_termination(self) -> None:
        env = _get_env(wrong_press_termination=True)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        # Simulate a wrong press by applying a generalized force on all the keys.
        env.physics.bind(env.task.piano.joints).qfrc_applied = 3.0

        # The episode should terminate in a single step.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # Failure, so discount should be 0.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 0.0)

    @absltest.skip("this observable is disabled")
    def test_steps_left_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)

        timestep = env.reset()
        self.assertEqual(timestep.observation["steps_left"], 1.0)

        for i in range(3):
            timestep = env.step(zero_action)
            self.assertAlmostEqual(
                timestep.observation["steps_left"], 1.0 - (i + 1) / 3
            )

    @parameterized.parameters(True, False)
    def test_fingering_reward_presence(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        env.step(zero_action)
        reward_terms = env.task.reward_fn.reward_terms

        if disable_fingering_reward:
            self.assertNotIn("fingering_reward", reward_terms)
        else:
            self.assertIn("fingering_reward", reward_terms)

    def test_grouped_reward_terms_split_dense_and_event_components(self) -> None:
        env = _get_env(onset_accuracy_reward_coef=1.0)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)

        env.reset()
        env.step(zero_action)

        reward_terms = env.task.reward_fn.reward_terms
        grouped_terms = env.task.get_grouped_reward_terms()

        expected_dense = sum(
            reward_terms.get(name, 0.0)
            for name in (
                "key_press_reward",
                "sustain_reward",
                "energy_reward",
                "fingering_reward",
                "ot_fingering_reward",
                "forearm_reward",
            )
        )
        expected_event = sum(
            reward_terms.get(name, 0.0)
            for name in ("onset_accuracy_reward", "velocity_reward")
        )

        self.assertAlmostEqual(grouped_terms["dense_reward"], expected_dense)
        self.assertAlmostEqual(grouped_terms["event_reward"], expected_event)
        self.assertAlmostEqual(
            grouped_terms["reward_total"], expected_dense + expected_event
        )

    # TODO(kevin): Add unit tests for individual reward components.
    # TODO(kevin): Add unit tests for augmentation / midi selection.


if __name__ == "__main__":
    absltest.main()
