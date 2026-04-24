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
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support

from robopianist.models.piano.midi_module import MAX_KEY_VEL as _MAX_KEY_VEL, QVEL_MIN as _QVEL_MIN
from robopianist.music.velocity_calibration import VelocityCalibration


def _fallback_score_key_metadata(task, t_idx: int, key_id: int) -> Dict[str, Optional[int | bool]]:
    """Builds score metadata from active-note trajectories when task helpers are absent."""
    if not (0 <= t_idx < len(task._notes)):
        return {
            "score_key_active": False,
            "gt_is_true_onset": False,
            "score_sustain": False,
            "gt_active_midi_vel": None,
            "gt_true_onset_midi_vel": None,
        }

    active_velocity_map = {note.key: int(note.velocity) for note in task._notes[t_idx]}
    prev_active_keys = set()
    if t_idx > 0:
        prev_active_keys = {note.key for note in task._notes[t_idx - 1]}
    true_onset_velocity_map = {
        key: velocity
        for key, velocity in active_velocity_map.items()
        if key not in prev_active_keys
    }
    return {
        "score_key_active": key_id in active_velocity_map,
        "gt_is_true_onset": key_id in true_onset_velocity_map,
        "score_sustain": bool(task._sustains[t_idx]) if 0 <= t_idx < len(task._sustains) else False,
        "gt_active_midi_vel": active_velocity_map.get(key_id),
        "gt_true_onset_midi_vel": true_onset_velocity_map.get(key_id),
    }


def _score_key_metadata(task, t_idx: int, key_id: int) -> Dict[str, Optional[int | bool]]:
    """Returns score metadata for a single key and timestep."""
    if hasattr(task, "get_score_key_metadata"):
        metadata = task.get_score_key_metadata(t_idx, key_id)
        return metadata._asdict()
    return _fallback_score_key_metadata(task, t_idx, key_id)


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

        self._velocity_calib = VelocityCalibration.load()
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

        # Detailed per-onset trace. The GT velocity fields refer to true GT onsets,
        # not merely score-active notes.
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
                self._episode_total_onsets += len(new_onsets)
                for key in new_onsets:
                    metadata = _score_key_metadata(task, t, int(key))
                    gt_vel = metadata["gt_true_onset_midi_vel"]
                    qvel = float(task.piano._onset_velocities[key])
                    robot_midi_vel = int(np.clip(
                        (qvel - _QVEL_MIN) / (_MAX_KEY_VEL - _QVEL_MIN) * 126, 0, 126
                    )) + 1
                    if gt_vel is None:
                        self._episode_unmatched_onsets += 1
                        self._episode_onset_trace.append({
                            "t_idx": t,
                            "key_id": int(key),
                            "robot_qvel": round(qvel, 4),
                            "robot_midi_vel": robot_midi_vel,
                            "gt_midi_vel": -1,
                            "score_active_midi_vel": (
                                int(metadata["gt_active_midi_vel"])
                                if metadata["gt_active_midi_vel"] is not None
                                else -1
                            ),
                            "needed_qvel": None,
                            "qvel_gap": None,
                            "error": None,
                            "score_key_active": bool(metadata["score_key_active"]),
                            "gt_is_true_onset": bool(metadata["gt_is_true_onset"]),
                            "score_sustain": bool(metadata["score_sustain"]),
                            "robot_new_onset": True,
                            "matched": False,
                        })
                        continue
                    needed_qvel = (float(gt_vel) - 1) / 126.0 * (_MAX_KEY_VEL - _QVEL_MIN) + _QVEL_MIN
                    self._episode_onset_trace.append({
                        "t_idx": t,
                        "key_id": int(key),
                        "robot_qvel": round(qvel, 4),
                        "robot_midi_vel": robot_midi_vel,
                        "gt_midi_vel": int(gt_vel),
                        "score_active_midi_vel": int(metadata["gt_active_midi_vel"]),
                        "needed_qvel": round(needed_qvel, 4),
                        "qvel_gap": round(qvel - needed_qvel, 4),
                        "error": robot_midi_vel - int(gt_vel),
                        "score_key_active": bool(metadata["score_key_active"]),
                        "gt_is_true_onset": bool(metadata["gt_is_true_onset"]),
                        "score_sustain": bool(metadata["score_sustain"]),
                        "robot_new_onset": True,
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

            self._all_onset_traces.append(list(self._episode_onset_trace))
            self._all_unmatched_onsets.append(self._episode_unmatched_onsets)
            self._all_total_onsets.append(self._episode_total_onsets)

            self._key_presses = []
            self._sustain_presses = []
            self._episode_onset_trace = []
            self._episode_unmatched_onsets = 0
            self._episode_total_onsets = 0
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._episode_onset_trace = []
        self._episode_unmatched_onsets = 0
        self._episode_total_onsets = 0
        return self._environment.reset()

    def get_velocity_metrics(self) -> Dict[str, float]:
        """Returns velocity statistics over the last `deque_size` episodes."""
        trace = [row for ep in self._all_onset_traces for row in ep]
        if not trace:
            return {}

        matched_rows = [row for row in trace if row["matched"]]
        if not matched_rows:
            return {}

        robot_arr = np.array([row["robot_midi_vel"] for row in matched_rows])
        gt_arr = np.array([row["gt_midi_vel"] for row in matched_rows])
        robot_qvel_arr = np.array([row["robot_qvel"] for row in matched_rows], dtype=np.float64)
        errors = robot_arr - gt_arr

        calib = self._velocity_calib
        robot_loud = np.array([calib.loudness_db(int(v)) for v in robot_arr])
        gt_loud = np.array([calib.loudness_db(int(v)) for v in gt_arr])
        loud_errors = robot_loud - gt_loud
        loud_corr = float(np.corrcoef(robot_loud, gt_loud)[0, 1]) if len(robot_loud) > 1 else float("nan")
        gt_std = float(np.std(gt_arr))
        dynamic_range_ratio = float(np.std(robot_arr)) / gt_std if gt_std > 0 else float("nan")

        # Perceptual Dynamics Score (PDS): harmonic mean of correlation and bias components.
        # s_corr = max(0, loud_corr); s_bias = exp(-|loud_bias| / 0.05)
        # PDS = 2 / (1/s_corr + 1/s_bias), or 0 if either component is 0.
        loud_bias = float(np.mean(loud_errors))
        s_corr = max(0.0, loud_corr) if not np.isnan(loud_corr) else 0.0
        s_bias = float(np.exp(-abs(loud_bias) / 0.05))
        pds = float(2.0 / (1.0 / s_corr + 1.0 / s_bias)) if s_corr > 0 and s_bias > 0 else 0.0

        return {
            "mean_robot_midi_vel": float(np.mean(robot_arr)),
            "std_robot_midi_vel": float(np.std(robot_arr)),
            "velocity_mae": float(np.mean(np.abs(errors))),
            "velocity_mse": float(np.mean(errors**2)),
            "velocity_bias": float(np.mean(errors)),  # positive = over-shooting GT
            "max_robot_onset_qvel": float(np.max(robot_qvel_arr)),
            "p90_robot_onset_qvel": float(np.percentile(robot_qvel_arr, 90)),
            "loudness_mae": float(np.mean(np.abs(loud_errors))),
            "loudness_bias": loud_bias,
            "loudness_correlation": loud_corr,
            "dynamic_range_ratio": dynamic_range_ratio,
            "perceptual_dynamics_score": pds,
        }

    def get_velocity_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (robot_midi_vels, gt_midi_vels) arrays for histogram plotting."""
        matched = [row for ep in self._all_onset_traces for row in ep if row["matched"]]
        robot = np.array([row["robot_midi_vel"] for row in matched], dtype=np.int32)
        gt = np.array([row["gt_midi_vel"] for row in matched], dtype=np.int32)
        return robot, gt

    def get_episode_velocity_trace(self) -> List[dict]:
        """Returns per-onset detail rows for the last episode(s).

        Each row has: t_idx, key_id, robot_qvel, robot_midi_vel, gt_midi_vel,
        error (robot - gt, None if unmatched), matched (bool),
        score_key_active, gt_is_true_onset, score_sustain, robot_new_onset.
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
