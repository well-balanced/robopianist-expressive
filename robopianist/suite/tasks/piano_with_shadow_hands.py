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

"""A task where two shadow hands must play a given MIDI file on a piano."""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.models.piano.midi_module import MAX_KEY_VEL as _MAX_KEY_VEL, QVEL_MIN as _QVEL_MIN
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05

# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3

# Key press reward coefficient.
_KEY_PRESS_REWARD_COEF = 1.0

# Velocity reward coefficient.
_VELOCITY_REWARD_COEF = 1.0
_VELOCITY_V2_HOLD_PENALTY_GRACE_STEPS = 2
_VELOCITY_V2_UNEXPECTED_HOLD_ONSET_PENALTY = 0.20
_VELOCITY_V2_PREMATURE_RELEASE_PENALTY = 0.10


# Transparency of fingertip geoms.
_FINGERTIP_ALPHA = 1.0

# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05


class ScoreKeyMetadata(NamedTuple):
    """Ground-truth score metadata for a single key at a single timestep."""

    score_key_active: bool
    gt_is_true_onset: bool
    score_sustain: bool
    gt_active_midi_vel: Optional[int]
    gt_true_onset_midi_vel: Optional[int]




class PianoWithShadowHands(base.PianoTask):
    def __init__(
        self,
        midi: midi_file.MidiFile,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_fingering_reward: bool = False,
        disable_forearm_reward: bool = False,
        disable_colorization: bool = False,
        disable_hand_collisions: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        energy_penalty_coef: float = _ENERGY_PENALTY_COEF,
        key_press_reward_coef: float = _KEY_PRESS_REWARD_COEF,
        randomize_hand_positions: bool = False,
        velocity_reward_coef: float = _VELOCITY_REWARD_COEF,
        disable_velocity_reward: bool = False,
        velocity_reward_version: str = "v1.0",
        use_velocity_reward_v2: bool = False,
        n_steps_velocity_lookahead: int = 3,
        compact_residual_obs: bool = False,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            disable_fingering_reward: If True, disables the shaping reward for
                fingering. This will also disable the colorization of the fingertips
                and corresponding keys. Note that if the MIDI file does not contain
                any fingering information, the fingering reward will also be disabled.
            disable_forearm_reward: If True, disables the shaping reward for the
                forearms.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            disable_hand_collisions: If True, disables collisions between the two hands.
            disable_velocity_reward: If True, disables the velocity reward term.
            velocity_reward_version: Legacy launch string for the velocity reward
                path. For backward compatibility, strings starting with `"v2"`
                select the current v2 reward and all other values select the
                current v1 reward.
            use_velocity_reward_v2: If True, forces the current v2 reward path.
                Kept for backward compatibility with older launch commands.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
            energy_penalty_coef: Coefficient for the energy penalty.
            key_press_reward_coef: Coefficient for the key press reward. Scales the
                key press reward term relative to other reward components.
            velocity_reward_coef: Coefficient for the velocity reward. Scales the
                velocity reward term relative to other reward components.
            randomize_hand_positions: If True, randomizes the initial position of the
                hands at the beginning of each episode.
            n_steps_velocity_lookahead: Number of timesteps to look ahead in the
                velocity goal observable. Independent of n_steps_lookahead.
        """
        super().__init__(arena=stage.Stage(), **kwargs)

        if trim_silence:
            midi = midi.trim_silence()
        self._midi = midi
        self._initial_midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._disable_fingering_reward = (
            disable_fingering_reward or not self._midi.has_fingering()
        )
        self._disable_forearm_reward = disable_forearm_reward
        self._velocity_reward_coef = velocity_reward_coef
        self._disable_velocity_reward = disable_velocity_reward
        self._use_velocity_reward_v2 = (
            use_velocity_reward_v2 or velocity_reward_version.startswith("v2")
        )
        self._n_steps_velocity_lookahead = n_steps_velocity_lookahead
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._disable_hand_collisions = disable_hand_collisions
        self._augmentations = augmentations
        self._energy_penalty_coef = energy_penalty_coef
        self._key_press_reward_coef = key_press_reward_coef
        self._compact_residual_obs = compact_residual_obs
        self._randomize_hand_positions = randomize_hand_positions
        self._score_active_velocity_maps: List[Dict[int, int]] = []
        self._score_true_onset_velocity_maps: List[Dict[int, int]] = []

        if not disable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        if disable_hand_collisions:
            self._disable_collisions_between_hands()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory()  # Important: call before adding observables.
        self._add_observables()
        self._set_rewards()

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
            energy_reward=self._compute_energy_reward,
        )
        if not self._disable_fingering_reward:
            self._reward_fn.add("fingering_reward", self._compute_fingering_reward)
        else:
            # use OT based fingering
            print('Fingering is unavailable. OT fingering reward is used.')
            self._reward_fn.add("ot_fingering_reward", self._compute_ot_fingering_reward)

        if not self._disable_forearm_reward:
            self._reward_fn.add("forearm_reward", self._compute_forearm_reward)

        if not self._disable_velocity_reward:
            self._reward_fn.add("velocity_reward", self._compute_velocity_reward)

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0
        self._goal_current: np.ndarray = np.zeros(
            self.piano.n_keys + 1, dtype=np.float64
        )
        self._prev_activation: np.ndarray = np.zeros(
            self.piano.n_keys, dtype=bool
        )
        self._velocity_goal_state: np.ndarray = np.zeros(
            (self._n_steps_velocity_lookahead + 1, self.piano.n_keys), dtype=np.float64
        )
        # v2 compares the current matched-onset mean against recent matched-onset
        # history, so these episode-local accumulators reset every episode.
        self._prev_velocity_reward_robot_mean: Optional[float] = None
        self._prev_velocity_reward_gt_mean: Optional[float] = None
        self._recent_velocity_reward_errors: List[float] = []

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
            self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        note_traj = midi_file.NoteTrajectory.from_midi(
            self._midi, self.control_timestep
        )
        note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._notes = note_traj.notes
        self._sustains = note_traj.sustains
        self._precompute_score_velocity_maps()

    def _precompute_score_velocity_maps(self) -> None:
        """Precompute active-note and true-onset velocity maps for the score."""
        self._score_active_velocity_maps = []
        self._score_true_onset_velocity_maps = []

        prev_active_keys = set()
        for notes in self._notes:
            active_velocity_map = {note.key: int(note.velocity) for note in notes}
            true_onset_velocity_map = {
                key: velocity
                for key, velocity in active_velocity_map.items()
                if key not in prev_active_keys
            }
            self._score_active_velocity_maps.append(active_velocity_map)
            self._score_true_onset_velocity_maps.append(true_onset_velocity_map)
            prev_active_keys = set(active_velocity_map)
        # v2 gives a small extra weight to unusually soft/loud score notes, so it
        # needs the piece-level center of the GT onset velocity distribution.
        true_onset_velocities = [
            velocity
            for velocity_map in self._score_true_onset_velocity_maps
            for velocity in velocity_map.values()
        ]
        if true_onset_velocities:
            self._piece_velocity_median = float(np.median(true_onset_velocities))
        else:
            self._piece_velocity_median = 64.0

    def _score_active_velocity_map(self, t_idx: int) -> Dict[int, int]:
        if 0 <= t_idx < len(self._score_active_velocity_maps):
            return self._score_active_velocity_maps[t_idx]
        return {}

    def _score_true_onset_velocity_map(self, t_idx: int) -> Dict[int, int]:
        if 0 <= t_idx < len(self._score_true_onset_velocity_maps):
            return self._score_true_onset_velocity_maps[t_idx]
        return {}


    def get_score_key_metadata(self, t_idx: int, key_id: int) -> ScoreKeyMetadata:
        """Returns score-side metadata for a key at a timestep.

        `score_key_active` answers whether the score expects the key to be active at
        this timestep, while `gt_is_true_onset` answers whether this timestep is the
        key's true GT onset. These are intentionally distinct because a long held note
        can be score-active without being a new onset.
        """
        active_velocity_map = self._score_active_velocity_map(t_idx)
        true_onset_velocity_map = self._score_true_onset_velocity_map(t_idx)
        score_sustain = bool(self._sustains[t_idx]) if 0 <= t_idx < len(self._sustains) else False
        return ScoreKeyMetadata(
            score_key_active=(key_id in active_velocity_map),
            gt_is_true_onset=(key_id in true_onset_velocity_map),
            score_sustain=score_sustain,
            gt_active_midi_vel=active_velocity_map.get(key_id),
            gt_true_onset_midi_vel=true_onset_velocity_map.get(key_id),
        )


    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()
        self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Applies the control to the hands and the sustain pedal to the piano."""
        self._prev_activation = self.piano.activation.copy()
        action_right, action_left = np.split(action[:-1], 2)
        self.right_hand.apply_action(physics, action_right, random_state)
        self.left_hand.apply_action(physics, action_left, random_state)
        self.piano.apply_sustain(physics, action[-1], random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        self._goal_current = self._goal_state[0]

        if not self._disable_fingering_reward:
            self._rh_keys_current = self._rh_keys
            self._lh_keys_current = self._lh_keys
            if not self._disable_colorization:
                self._colorize_keys(physics)

        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

    def get_reward(self, physics: mjcf.Physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        hands_spec = spec_utils.merge_specs([right_spec, left_spec])
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hands_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([hands_spec, sustain_spec])

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    # Helper methods.

    def _compute_forearm_reward(self, physics: mjcf.Physics) -> float:
        """Reward for not colliding the forearms."""
        if collision_utils.has_collision(
            physics,
            [g.full_identifier for g in self.right_hand.root_body.geom],
            [g.full_identifier for g in self.left_hand.root_body.geom],
        ):
            return 0.0
        return 0.5

    def _compute_sustain_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the sustain pedal at the right time."""
        del physics  # Unused.
        return tolerance(
            self._goal_current[-1] - self.piano.sustain_activation[0],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing energy."""
        rew = 0.0
        for hand in [self.right_hand, self.left_hand]:
            power = hand.observables.actuators_power(physics).copy()
            rew -= self._energy_penalty_coef * np.sum(power)
        return rew


    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        # It's possible we have no keys to press at this timestep, so we need to check
        # that `on` is not empty.
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = np.flatnonzero(1 - self._goal_current[:-1])
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))
        return self._key_press_reward_coef * rew

    def _compute_velocity_reward(self, physics: mjcf.Physics) -> float:
        """Compute the current velocity reward.

        The codebase only keeps the latest v1 and v2 implementations in-tree.
        Older variants live in git history, not as runtime branches.
        """
        if self._use_velocity_reward_v2:
            return self._compute_velocity_reward_v2(physics)
        return self._compute_velocity_reward_v1(physics)

    def _compute_velocity_reward_v1(self, physics: mjcf.Physics) -> float:
        """Simple matched-onset absolute velocity reward.

        Flow:
        1. detect robot onsets with `activation & ~prev_activation`
        2. look up only score *true onsets* at the same timestep
        3. give 0 on non-onset steps and unmatched robot onsets
        4. score matched onsets with a sharp `tolerance()` on MIDI velocity

        This keeps v1 intentionally simple:
        - no contour term
        - no running bias term
        - no piece-level extreme weighting

        Compared with the older v1, this removes the overly flat
        `1 + loudness_calib.reward(...)` shaping and aligns the reward target
        with the evaluation semantics, which also operate on matched true
        onsets rather than all score-active keys.
        """
        del physics  # Unused.
        new_onsets = self.piano.activation & ~self._prev_activation
        if not new_onsets.any():
            return 0.0

        t = self._t_idx - 1
        gt_true_onset_velocity_map = self._score_true_onset_velocity_map(t)
        rewards = []
        for key in np.flatnonzero(new_onsets):
            gt_vel = gt_true_onset_velocity_map.get(int(key))
            if gt_vel is None:
                rewards.append(0.0)
                continue
            robot_midi_vel = (
                int(
                    np.clip(
                        (self.piano._onset_velocities[key] - _QVEL_MIN)
                        / (_MAX_KEY_VEL - _QVEL_MIN)
                        * 126,
                        0,
                        126,
                    )
                )
                + 1
            )

            lo = max(1, gt_vel - 3)
            hi = min(127, gt_vel + 3)
            accuracy = tolerance(
                robot_midi_vel,
                bounds=(lo, hi),
                margin=20,
                sigmoid="gaussian",
                value_at_margin=0.05,
            )
            rewards.append(2.0 * float(accuracy) - 1.0)

        return self._velocity_reward_coef * float(np.mean(rewards))

    def _compute_velocity_reward_v2(self, physics: mjcf.Physics) -> float:
        """Matched-onset reward with hold-stability penalties.

        Flow:
        1. detect robot onsets with `activation & ~prev_activation`
        2. detect releases with `~activation & prev_activation`
        3. reward matched GT true onsets for velocity accuracy
        4. penalize an onset if the score says the key should already be held
        5. penalize a release if the score still expects the key to stay active
        6. add local contour and short-window bias terms on matched onsets only

        The first couple of control steps are exempt from hold penalties.
        Twinkle starts with notes that are already active because of
        `initial_buffer_time`, so penalizing those steps directly makes the
        reward collapse before the policy even reaches the main phrase.
        """
        del physics  # Unused.

        activation = self.piano.activation
        new_onsets = activation & ~self._prev_activation
        releases = ~activation & self._prev_activation

        t = self._t_idx - 1
        penalize_hold_stability = t >= _VELOCITY_V2_HOLD_PENALTY_GRACE_STEPS
        gt_active_velocity_map = self._score_active_velocity_map(t)
        gt_true_onset_velocity_map = self._score_true_onset_velocity_map(t)

        matched_robot_vels: List[int] = []
        matched_gt_vels: List[int] = []
        abs_rewards: List[float] = []
        unexpected_hold_onset_count = 0

        for key in np.flatnonzero(new_onsets):
            key_id = int(key)
            gt_vel = gt_true_onset_velocity_map.get(key_id)
            if gt_vel is None:
                # This branch captures the exact failure mode we saw in traces:
                # the score still wants the note to be active, but the robot
                # created a fresh onset instead of smoothly keeping it held.
                if penalize_hold_stability and key_id in gt_active_velocity_map:
                    unexpected_hold_onset_count += 1
                continue

            robot_qvel = float(self.piano._onset_velocities[key])
            robot_midi_vel = (
                int(
                    np.clip(
                        (robot_qvel - _QVEL_MIN) / (_MAX_KEY_VEL - _QVEL_MIN) * 126,
                        0,
                        126,
                    )
                )
                + 1
            )

            lo = max(1, gt_vel - 3)
            hi = min(127, gt_vel + 3)
            abs_accuracy = tolerance(
                robot_midi_vel,
                bounds=(lo, hi),
                margin=20,
                sigmoid="gaussian",
                value_at_margin=0.05,
            )

            matched_robot_vels.append(robot_midi_vel)
            matched_gt_vels.append(gt_vel)
            abs_rewards.append(2.0 * float(abs_accuracy) - 1.0)

        premature_release_count = (
            sum(
                1
                for key in np.flatnonzero(releases)
                if int(key) in gt_active_velocity_map
            )
            if penalize_hold_stability
            else 0
        )

        abs_reward = float(np.mean(abs_rewards)) if abs_rewards else 0.0

        # Chords produce multiple onsets in one step, so v2 first compresses that
        # step to a mean velocity before comparing it against recent history.
        if not matched_robot_vels:
            contour_reward = 0.0
            bias_reward = 0.0
            extreme_weight = 1.0
        else:
            step_robot_mean = float(np.mean(matched_robot_vels))
            step_gt_mean = float(np.mean(matched_gt_vels))
            step_error = step_robot_mean - step_gt_mean

            if (
                self._prev_velocity_reward_robot_mean is None
                or self._prev_velocity_reward_gt_mean is None
            ):
                contour_reward = 0.0
            else:
                robot_delta = step_robot_mean - self._prev_velocity_reward_robot_mean
                gt_delta = step_gt_mean - self._prev_velocity_reward_gt_mean
                if abs(gt_delta) < 4.0:
                    contour_reward = 0.0
                else:
                    contour_reward = float(
                        np.tanh(robot_delta / 8.0) * np.tanh(gt_delta / 8.0)
                    )

            if len(self._recent_velocity_reward_errors) < 2:
                bias_reward = 0.0
            else:
                recent_errors = self._recent_velocity_reward_errors + [step_error]
                recent_bias = float(np.mean(recent_errors))
                bias_reward = -min(abs(recent_bias) / 15.0, 1.0)

            extreme_weight = float(
                np.mean(
                    [
                        1.0
                        + 0.5
                        * min(abs(gt_vel - self._piece_velocity_median) / 20.0, 1.0)
                        for gt_vel in matched_gt_vels
                    ]
                )
            )

            self._prev_velocity_reward_robot_mean = step_robot_mean
            self._prev_velocity_reward_gt_mean = step_gt_mean
            self._recent_velocity_reward_errors.append(step_error)
            if len(self._recent_velocity_reward_errors) > 4:
                self._recent_velocity_reward_errors.pop(0)

        hold_stability_penalty = (
            _VELOCITY_V2_UNEXPECTED_HOLD_ONSET_PENALTY * unexpected_hold_onset_count
            + _VELOCITY_V2_PREMATURE_RELEASE_PENALTY * premature_release_count
        )

        raw_reward = extreme_weight * (
            abs_reward
            + 0.35 * contour_reward
            + 0.15 * bias_reward
        ) - hold_stability_penalty
        clipped_reward = float(np.clip(raw_reward, -1.5, 1.5))
        return self._velocity_reward_coef * clipped_reward

    def _compute_fingering_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""

        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances

        distances = _distance_finger_to_key(self._rh_keys_current, self.right_hand)
        distances += _distance_finger_to_key(self._lh_keys_current, self.left_hand)

        # Case where there are no keys to press at this timestep.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))

    def _compute_ot_fingering_reward(self, physics: mjcf.Physics) -> float:
        """ OT reward calculation from RP1M https://arxiv.org/abs/2408.11048 """
        # calcuate fingertip positions
        fingertip_pos = [physics.bind(finger).xpos.copy() for finger in self.left_hand.fingertip_sites]
        fingertip_pos += [physics.bind(finger).xpos.copy() for finger in self.right_hand.fingertip_sites]
        
        # calcuate the positions of piano keys to press.
        keys_to_press = np.flatnonzero(self._goal_current[:-1]) # keys to press
        # if no key is pressed
        if keys_to_press.shape[0] == 0:
            return 1.

        # calculate key pos
        key_pos = []
        for key in keys_to_press:
            key_geom = self.piano.keys[key].geom[0]
            key_geom_pos = physics.bind(key_geom).xpos.copy()
            key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
            key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
            key_pos.append(key_geom_pos.copy())

        # calcualte the distance between keys and fingers
        dist = np.full((len(fingertip_pos), len(key_pos)), 100.)
        for i, finger in enumerate(fingertip_pos):
            for j, key in enumerate(key_pos):
                dist[i, j] = np.linalg.norm(key - finger)
        
        # calculate the shortest distance
        row_ind, col_ind = linear_sum_assignment(dist)
        dist = dist[row_ind, col_ind]
        rews = tolerance(
            dist,
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))        

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _update_velocity_goal_state(self) -> None:
        if self._t_idx == len(self._notes):
            return
        self._velocity_goal_state = np.zeros(
            (self._n_steps_velocity_lookahead + 1, self.piano.n_keys),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_velocity_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            for note in self._notes[t]:
                self._velocity_goal_state[i, note.key] = note.velocity / 127.0

    def _get_velocity_scaled_goal_state(self) -> np.ndarray:
        """Goal state with velocity scaling instead of binary 1.0.

        Same shape as the regular goal state: (n_steps_lookahead+1, n_keys+1).
        Where the regular goal has 1.0 for a pressed key, this has velocity/127.
        The sustain dimension (last) stays binary, as sustain has no velocity.
        Returns zeros if the episode is ending.
        """
        n = self._n_steps_lookahead + 1
        result = np.zeros((n, self.piano.n_keys + 1), dtype=np.float64)
        if self._t_idx == len(self._notes):
            return result
        t_start = self._t_idx
        t_end = min(t_start + n, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            for note in self._notes[t]:
                result[i, note.key] = note.velocity / 127.0
            result[i, -1] = self._sustains[t]
        return result

    def _update_fingering_state(self) -> None:
        if self._t_idx == len(self._notes):
            return

        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]

        # Split fingering into right and left hand.
        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        # For each hand, set the finger to 1 if it is used and 0 otherwise.
        self._fingering_state = np.zeros((2, 5), dtype=np.float64)
        for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
            for key, mjcf_fingering in keys:
                self._fingering_state[hand, mjcf_fingering] = 1.0

    def _get_fingering_velocity_state(self) -> np.ndarray:
        """Returns per-finger target velocities for the current timestep."""
        state = np.zeros((2, 5), dtype=np.float64)
        if self._t_idx == len(self._notes):
            return state

        for note in self._notes[self._t_idx]:
            finger = note.fingering
            if finger < 0:
                continue
            if finger < 5:
                hand = 0
                hand_finger = finger
            else:
                hand = 1
                hand_finger = finger - 5

            state[hand, hand_finger] = max(state[hand, hand_finger], note.velocity / 127.0)
        return state

    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            "joints_pos",
            # NOTE(kevin): This observable was previously enabled but it is redundant
            # since it is encoded in the joint positions, specifically via the forearm
            # slider joints (which are in units of meters).
            # "position",
        ]
        enabled_observables.append("joints_vel")
        for hand in [self.right_hand, self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        # This returns the current state of the piano keys.
        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True

        # Goal state: binary 1.0 entries replaced by velocity/127 (scaled_goal).
        # _goal_state stays binary for reward computation.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._get_velocity_scaled_goal_state().ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        # This adds fingering information for the current timestep.
        def _get_fingering_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = not self._disable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

        def _get_fingering_velocity_state(physics) -> np.ndarray:
            del physics  # Unused.
            return self._get_fingering_velocity_state().ravel()

        fingering_velocity_observable = observable.Generic(_get_fingering_velocity_state)
        fingering_velocity_observable.enabled = False
        self._task_observables["fingering_velocity"] = fingering_velocity_observable

        # Compact residual obs: velocity lookahead + piano state + hand kinematics.
        # Appended at the END of the obs vector so obs[:base_obs_dim] stays intact.
        # Shape: (n_steps_velocity_lookahead+1)*88 + 88 + 1 + 23*4 = 621 (default 4-step).
        if self._compact_residual_obs:
            def _get_compact_residual_obs(physics) -> np.ndarray:
                self._update_velocity_goal_state()
                vel = self._velocity_goal_state.ravel()
                piano_state = self.piano._state.copy()
                sustain = self.piano._sustain_state.copy()
                rh_pos = physics.bind(self.right_hand.joints).qpos.copy()
                lh_pos = physics.bind(self.left_hand.joints).qpos.copy()
                rh_vel = physics.bind(self.right_hand.joints).qvel.copy()
                lh_vel = physics.bind(self.left_hand.joints).qvel.copy()
                return np.concatenate([vel, piano_state, sustain, rh_pos, lh_pos, rh_vel, lh_vel])

            compact_obs = observable.Generic(_get_compact_residual_obs)
            compact_obs.enabled = True
            self._task_observables["compact_residual_obs"] = compact_obs

    def _colorize_fingertips(self) -> None:
        """Colorize the fingertips of the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for i, body in enumerate(hand.fingertip_bodies):
                color = hand_consts.FINGERTIP_COLORS[i] + (_FINGERTIP_ALPHA,)
                for geom in body.find_all("geom"):
                    if geom.dclass.dclass == "plastic_visual":
                        geom.rgba = color
                # Also color the fingertip sites.
                hand.fingertip_sites[i].rgba = color

    def _colorize_keys(self, physics) -> None:
        """Colorize the keys by the corresponding fingertip color."""
        for hand, keys in zip(
            [self.right_hand, self.left_hand],
            [self._rh_keys_current, self._lh_keys_current],
        ):
            for key, mjcf_fingering in keys:
                key_geom = self.piano.keys[key].geom[0]
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                if not self.piano.activation[key]:
                    physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (
                        1.0,
                    )

    def _disable_collisions_between_hands(self) -> None:
        """Disable collisions between the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for geom in hand.mjcf_model.find_all("geom"):
                # If both hands have the same contype and conaffinity, then they can't
                # collide. They can still collide with the piano since the piano has
                # contype 0 and conaffinity 1. Lastly, we make sure we're not changing
                # the contype and conaffinity of the hand geoms that are already
                # disabled (i.e., the visual geoms).
                commit_defaults(geom, ["contype", "conaffinity"])
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.conaffinity = 0
                geom.contype = 1

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Randomize the initial position of the hands."""
        if not self._randomize_hand_positions:
            return
        offset = random_state.uniform(low=-_POSITION_OFFSET, high=_POSITION_OFFSET)
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, offset, 0))
