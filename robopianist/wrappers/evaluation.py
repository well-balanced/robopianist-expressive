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

"""A wrapper for tracking episode statistics pertaining to music performance.

TODO(kevin):
- Look into `mir_eval` for metrics.
- Should sustain be a separate metric or should it just be applied to the note sequence
    as a whole?
"""

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Sequence, Tuple

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support

from robopianist.models.piano.midi_module import MAX_KEY_VEL as _MAX_KEY_VEL


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(self, environment: dm_env.Environment, deque_size: int = 1) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []
        self._sustain_presses: List[np.ndarray] = []

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

        # Sustain metrics.
        self._sustain_precisions: Deque[float] = deque(maxlen=deque_size)
        self._sustain_recalls: Deque[float] = deque(maxlen=deque_size)
        self._sustain_f1s: Deque[float] = deque(maxlen=deque_size)

        # Velocity tracking (per onset).
        self._episode_robot_vels: List[int] = []
        self._episode_gt_vels: List[int] = []
        self._episode_robot_qvels: List[float] = []
        self._all_robot_vels: Deque[List[int]] = deque(maxlen=deque_size)
        self._all_gt_vels: Deque[List[int]] = deque(maxlen=deque_size)
        self._all_robot_qvels: Deque[List[float]] = deque(maxlen=deque_size)

        # Detailed per-onset trace (key_id, t_idx, qvel, midi vels, error, match flag).
        self._episode_onset_trace: List[dict] = []
        self._all_onset_traces: Deque[List[dict]] = deque(maxlen=deque_size)
        # Count of robot onsets that had no matching GT note (wrong key / timing mismatch).
        self._episode_unmatched_onsets: int = 0
        self._all_unmatched_onsets: Deque[int] = deque(maxlen=deque_size)
        self._episode_total_onsets: int = 0
        self._all_total_onsets: Deque[int] = deque(maxlen=deque_size)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = self._environment.task.piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))

        # Velocity tracking: record robot and GT MIDI velocity at each new onset.
        task = self._environment.task
        new_onsets = np.flatnonzero(key_activation & ~task._prev_activation)
        if new_onsets.size > 0:
            t = task._t_idx - 1
            if 0 <= t < len(task._notes):
                gt_vel_map = {note.key: note.velocity for note in task._notes[t]}
                self._episode_total_onsets += len(new_onsets)
                for key in new_onsets:
                    gt_vel = gt_vel_map.get(int(key))
                    qvel = float(task.piano._onset_velocities[key])
                    robot_midi_vel = int(np.clip(qvel / _MAX_KEY_VEL * 126, 0, 126)) + 1
                    if gt_vel is None:
                        self._episode_unmatched_onsets += 1
                        self._episode_onset_trace.append({
                            "t_idx": t,
                            "key_id": int(key),
                            "robot_qvel": round(qvel, 4),
                            "robot_midi_vel": robot_midi_vel,
                            "gt_midi_vel": -1,
                            "needed_qvel": None,
                            "qvel_gap": None,
                            "error": None,
                            "matched": False,
                        })
                        continue
                    needed_qvel = float(gt_vel) / 127.0 * _MAX_KEY_VEL
                    self._episode_robot_vels.append(robot_midi_vel)
                    self._episode_gt_vels.append(int(gt_vel))
                    self._episode_robot_qvels.append(qvel)
                    self._episode_onset_trace.append({
                        "t_idx": t,
                        "key_id": int(key),
                        "robot_qvel": round(qvel, 4),
                        "robot_midi_vel": robot_midi_vel,
                        "gt_midi_vel": int(gt_vel),
                        "needed_qvel": round(needed_qvel, 4),
                        "qvel_gap": round(qvel - needed_qvel, 4),
                        "error": robot_midi_vel - int(gt_vel),
                        "matched": True,
                    })

        if timestep.last():
            key_press_metrics = self._compute_key_press_metrics()
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            sustain_metrics = self._compute_sustain_metrics()
            self._sustain_precisions.append(sustain_metrics.precision)
            self._sustain_recalls.append(sustain_metrics.recall)
            self._sustain_f1s.append(sustain_metrics.f1)

            self._all_robot_vels.append(list(self._episode_robot_vels))
            self._all_gt_vels.append(list(self._episode_gt_vels))
            self._all_robot_qvels.append(list(self._episode_robot_qvels))
            self._all_onset_traces.append(list(self._episode_onset_trace))
            self._all_unmatched_onsets.append(self._episode_unmatched_onsets)
            self._all_total_onsets.append(self._episode_total_onsets)

            self._key_presses = []
            self._sustain_presses = []
            self._episode_robot_vels = []
            self._episode_gt_vels = []
            self._episode_robot_qvels = []
            self._episode_onset_trace = []
            self._episode_unmatched_onsets = 0
            self._episode_total_onsets = 0
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._episode_robot_vels = []
        self._episode_gt_vels = []
        self._episode_robot_qvels = []
        self._episode_onset_trace = []
        self._episode_unmatched_onsets = 0
        self._episode_total_onsets = 0
        return self._environment.reset()

    def get_velocity_metrics(self) -> Dict[str, float]:
        """Returns velocity statistics over the last `deque_size` episodes."""
        robot_vels = [v for ep in self._all_robot_vels for v in ep]
        gt_vels = [v for ep in self._all_gt_vels for v in ep]
        if not robot_vels:
            return {}
        robot_arr = np.array(robot_vels)
        gt_arr = np.array(gt_vels)
        robot_qvels = [v for ep in self._all_robot_qvels for v in ep]
        robot_qvel_arr = np.array(robot_qvels) if robot_qvels else np.array([0.0])
        errors = robot_arr - gt_arr
        total = sum(self._all_total_onsets) if self._all_total_onsets else 0
        return {
            "mean_robot_midi_vel": float(np.mean(robot_arr)),
            "std_robot_midi_vel": float(np.std(robot_arr)),
            "mean_gt_midi_vel": float(np.mean(gt_arr)),
            "velocity_mae": float(np.mean(np.abs(errors))),
            "velocity_bias": float(np.mean(errors)),  # positive = over-shooting GT
            "max_robot_onset_qvel": float(np.max(robot_qvel_arr)),
            "p90_robot_onset_qvel": float(np.percentile(robot_qvel_arr, 90)),
            "mean_robot_onset_qvel": float(np.mean(robot_qvel_arr)),
            "onset_match_rate": float(robot_arr.size / total) if total > 0 else 0.0,
        }

    def get_velocity_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (robot_midi_vels, gt_midi_vels) arrays for histogram plotting."""
        robot_vels = [v for ep in self._all_robot_vels for v in ep]
        gt_vels = [v for ep in self._all_gt_vels for v in ep]
        return np.array(robot_vels, dtype=np.int32), np.array(gt_vels, dtype=np.int32)

    def get_episode_velocity_trace(self) -> List[dict]:
        """Returns per-onset detail rows for the last episode(s).

        Each row has: t_idx, key_id, robot_qvel, robot_midi_vel, gt_midi_vel,
        error (robot - gt, None if unmatched), matched (bool).
        Suitable for logging as a wandb.Table.
        """
        return [row for ep in self._all_onset_traces for row in ep]

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean precision/recall/F1 over the last `deque_size` episodes."""
        if not self._key_press_precisions:
            raise ValueError("No episode metrics available yet.")

        def _mean(seq: Sequence[float]) -> float:
            return sum(seq) / len(seq)

        return {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
            "sustain_precision": _mean(self._sustain_precisions),
            "sustain_recall": _mean(self._sustain_recalls),
            "sustain_f1": _mean(self._sustain_f1s),
        }

    # Helper methods.

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for key presses over the episode."""
        # Get the ground truth key presses.
        note_seq = self._environment.task._notes
        ground_truth = []
        for notes in note_seq:
            presses = np.zeros((self._environment.task.piano.n_keys,), dtype=np.float64)
            keys = [note.key for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # Deal with the case where the episode gets truncated due to a failure. In this
        # case, the length of the key presses will be less than or equal to the length
        # of the ground truth.
        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._key_presses)]

        assert len(ground_truth) == len(self._key_presses)

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._key_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for sustain presses over the episode."""
        # Get the ground truth sustain presses.
        ground_truth = [
            np.atleast_1d(v).astype(float) for v in self._environment.task._sustains
        ]

        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._sustain_presses)]

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._sustain_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)
