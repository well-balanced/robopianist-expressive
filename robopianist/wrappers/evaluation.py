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
from robopianist.music import constants as music_consts
from robopianist.music import midi_file, midi_message, synthesizer


_AUDIO_SAMPLE_RATE = music_consts.SAMPLING_RATE
_AUDIO_ENV_WINDOW_SECONDS = 0.05
_AUDIO_ENV_HOP_SECONDS = 0.025
_INT16_MAX = float(np.iinfo(np.int16).max)


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


def _true_onset_velocity_map(task, t_idx: int) -> Dict[int, int]:
    """Returns the GT true-onset velocity map at a timestep."""
    if hasattr(task, "_score_true_onset_velocity_map"):
        return dict(task._score_true_onset_velocity_map(t_idx))

    if hasattr(task, "get_score_key_metadata") and hasattr(task, "piano"):
        true_onset_velocity_map = {}
        for key_id in range(task.piano.n_keys):
            metadata = task.get_score_key_metadata(t_idx, key_id)
            gt_vel = metadata.gt_true_onset_midi_vel
            if gt_vel is not None:
                true_onset_velocity_map[int(key_id)] = int(gt_vel)
        return true_onset_velocity_map

    if not (0 <= t_idx < len(task._notes)):
        return {}

    active_velocity_map = {note.key: int(note.velocity) for note in task._notes[t_idx]}
    prev_active_keys = set()
    if t_idx > 0:
        prev_active_keys = {note.key for note in task._notes[t_idx - 1]}
    return {
        key: velocity
        for key, velocity in active_velocity_map.items()
        if key not in prev_active_keys
    }


def _event_priority(event: midi_message.MidiMessage) -> int:
    if isinstance(event, midi_message.NoteOff):
        return 0
    if isinstance(event, midi_message.SustainOff):
        return 1
    if isinstance(event, midi_message.SustainOn):
        return 2
    return 3


def _clone_midi_event(event: midi_message.MidiMessage) -> midi_message.MidiMessage:
    if isinstance(event, midi_message.NoteOn):
        return midi_message.NoteOn(
            note=event.note, velocity=event.velocity, time=float(event.time)
        )
    if isinstance(event, midi_message.NoteOff):
        return midi_message.NoteOff(note=event.note, time=float(event.time))
    if isinstance(event, midi_message.SustainOn):
        return midi_message.SustainOn(time=float(event.time))
    if isinstance(event, midi_message.SustainOff):
        return midi_message.SustainOff(time=float(event.time))
    raise TypeError(f"Unsupported MIDI event type: {type(event)!r}")


def _serialize_midi_event(event: midi_message.MidiMessage) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {
        "event_type": type(event).__name__,
        "time": float(event.time),
    }
    if hasattr(event, "note"):
        row["note"] = int(event.note)
    if hasattr(event, "velocity"):
        row["velocity"] = int(event.velocity)
    return row


def _sort_midi_events(
    events: Sequence[midi_message.MidiMessage],
) -> List[midi_message.MidiMessage]:
    return sorted(
        (_clone_midi_event(event) for event in events),
        key=lambda event: (float(event.time), _event_priority(event)),
    )


def _note_to_midi_number(note) -> int:
    if hasattr(note, "number"):
        return int(note.number)
    return midi_file.key_number_to_midi_number(int(note.key))


def _score_midi_events_from_trajectory(task) -> List[midi_message.MidiMessage]:
    """Build GT MIDI events from the discretized target trajectory."""
    if not hasattr(task, "_notes") or not hasattr(task, "_sustains"):
        return []

    dt = float(task.control_timestep)
    prev_active: Dict[int, int] = {}
    prev_sustain = False
    events: List[midi_message.MidiMessage] = []

    for t_idx, notes in enumerate(task._notes):
        current_active = {
            _note_to_midi_number(note): int(note.velocity) for note in notes
        }
        current_time = float(t_idx * dt)

        for midi_number in sorted(prev_active.keys() - current_active.keys()):
            events.append(midi_message.NoteOff(note=midi_number, time=current_time))

        for midi_number in sorted(current_active.keys() - prev_active.keys()):
            events.append(
                midi_message.NoteOn(
                    note=midi_number,
                    velocity=current_active[midi_number],
                    time=current_time,
                )
            )

        current_sustain = bool(task._sustains[t_idx]) if t_idx < len(task._sustains) else False
        if current_sustain and not prev_sustain:
            events.append(midi_message.SustainOn(time=current_time))
        if prev_sustain and not current_sustain:
            events.append(midi_message.SustainOff(time=current_time))

        prev_active = current_active
        prev_sustain = current_sustain

    final_time = float(len(task._notes) * dt)
    for midi_number in sorted(prev_active):
        events.append(midi_message.NoteOff(note=midi_number, time=final_time))
    if prev_sustain:
        events.append(midi_message.SustainOff(time=final_time))
    return _sort_midi_events(events)


def _robot_midi_events_from_episode(
    *,
    key_presses: Sequence[np.ndarray],
    sustain_presses: Sequence[np.ndarray],
    onset_trace: Sequence[dict],
    control_timestep: float,
) -> List[midi_message.MidiMessage]:
    """Reconstruct robot MIDI events from episode state traces."""
    onset_velocity_by_step: Dict[tuple[int, int], int] = {}
    for row in onset_trace:
        if row.get("robot_new_onset"):
            onset_velocity_by_step[(int(row["t_idx"]), int(row["key_id"]))] = int(
                row["robot_midi_vel"]
            )

    prev_active = set()
    prev_sustain = False
    events: List[midi_message.MidiMessage] = []

    for t_idx, activation in enumerate(key_presses):
        current_active = set(np.flatnonzero(np.asarray(activation) > 0.5))
        current_time = float(t_idx * control_timestep)

        for key_id in sorted(prev_active - current_active):
            events.append(
                midi_message.NoteOff(
                    note=midi_file.key_number_to_midi_number(int(key_id)),
                    time=current_time,
                )
            )

        for key_id in sorted(current_active - prev_active):
            velocity = onset_velocity_by_step.get((t_idx, int(key_id)), 127)
            events.append(
                midi_message.NoteOn(
                    note=midi_file.key_number_to_midi_number(int(key_id)),
                    velocity=int(velocity),
                    time=current_time,
                )
            )

        current_sustain = False
        if t_idx < len(sustain_presses):
            current_sustain = bool(np.asarray(sustain_presses[t_idx]).astype(bool).any())
        if current_sustain and not prev_sustain:
            events.append(midi_message.SustainOn(time=current_time))
        if prev_sustain and not current_sustain:
            events.append(midi_message.SustainOff(time=current_time))

        prev_active = current_active
        prev_sustain = current_sustain

    final_time = float(len(key_presses) * control_timestep)
    for key_id in sorted(prev_active):
        events.append(
            midi_message.NoteOff(
                note=midi_file.key_number_to_midi_number(int(key_id)),
                time=final_time,
            )
        )
    if prev_sustain:
        events.append(midi_message.SustainOff(time=final_time))
    return _sort_midi_events(events)


def _align_waveforms(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if len(a) < max_len:
        a = np.pad(a, (0, max_len - len(a)))
    if len(b) < max_len:
        b = np.pad(b, (0, max_len - len(b)))
    return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)


def _rms_envelope(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float = _AUDIO_ENV_WINDOW_SECONDS,
    hop_seconds: float = _AUDIO_ENV_HOP_SECONDS,
) -> np.ndarray:
    if waveform.size == 0:
        return np.zeros((0,), dtype=np.float32)

    frame = max(1, int(round(window_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    if waveform.size < frame:
        waveform = np.pad(waveform, (0, frame - waveform.size))

    values = []
    last_start = max(0, waveform.size - frame)
    for start in range(0, last_start + 1, hop):
        chunk = waveform[start : start + frame]
        values.append(float(np.sqrt(np.mean(chunk**2))))
    if last_start % hop != 0:
        chunk = waveform[last_start : last_start + frame]
        values.append(float(np.sqrt(np.mean(chunk**2))))
    return np.asarray(values, dtype=np.float32)


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

    def __init__(
        self,
        environment: dm_env.Environment,
        deque_size: int = 1,
        success_criteria: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(environment)

        self._audio_sample_rate = _AUDIO_SAMPLE_RATE
        self._audio_synth = synthesizer.Synthesizer(sample_rate=self._audio_sample_rate)
        self._success_criteria = (
            dict(success_criteria) if success_criteria is not None else None
        )
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

        # Onset metrics.
        self._episode_gt_onsets: List[np.ndarray] = []
        self._episode_robot_onsets: List[np.ndarray] = []
        self._onset_precisions: Deque[float] = deque(maxlen=deque_size)
        self._onset_recalls: Deque[float] = deque(maxlen=deque_size)
        self._onset_f1s: Deque[float] = deque(maxlen=deque_size)

        # Detailed per-onset trace. The GT velocity fields refer to true GT onsets,
        # not merely score-active notes.
        self._episode_onset_trace: List[dict] = []
        self._all_onset_traces: Deque[List[dict]] = deque(maxlen=deque_size)
        # Count of robot onsets that had no matching GT note (wrong key / timing mismatch).
        self._episode_unmatched_onsets: int = 0
        self._all_unmatched_onsets: Deque[int] = deque(maxlen=deque_size)
        self._episode_total_onsets: int = 0
        self._all_total_onsets: Deque[int] = deque(maxlen=deque_size)
        self._episode_metric_rows: Deque[Dict[str, float]] = deque(maxlen=deque_size)
        self._latest_audio_renderings: Dict[str, object] = {}

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = self._environment.task.piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))

        # Velocity tracking: record robot and GT MIDI velocity at each new onset.
        task = self._environment.task
        new_onsets = np.flatnonzero(key_activation & ~task._prev_activation)
        t = task._t_idx - 1
        robot_onset_vec = np.zeros((task.piano.n_keys,), dtype=np.float64)
        if new_onsets.size > 0:
            robot_onset_vec[new_onsets] = 1.0
        gt_onset_vec = np.zeros((task.piano.n_keys,), dtype=np.float64)
        gt_true_onset_velocity_map = _true_onset_velocity_map(task, t)
        if gt_true_onset_velocity_map:
            gt_onset_vec[list(gt_true_onset_velocity_map.keys())] = 1.0
        self._episode_robot_onsets.append(robot_onset_vec)
        self._episode_gt_onsets.append(gt_onset_vec)

        if new_onsets.size > 0:
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

            onset_metrics = self._compute_onset_metrics()
            self._onset_precisions.append(onset_metrics.precision)
            self._onset_recalls.append(onset_metrics.recall)
            self._onset_f1s.append(onset_metrics.f1)
            episode_row = self._build_episode_metric_row(
                key_press_metrics=key_press_metrics,
                sustain_metrics=sustain_metrics,
                onset_metrics=onset_metrics,
            )
            self._episode_metric_rows.append(episode_row)

            self._all_onset_traces.append(list(self._episode_onset_trace))
            self._all_unmatched_onsets.append(self._episode_unmatched_onsets)
            self._all_total_onsets.append(self._episode_total_onsets)

            self._key_presses = []
            self._sustain_presses = []
            self._episode_gt_onsets = []
            self._episode_robot_onsets = []
            self._episode_onset_trace = []
            self._episode_unmatched_onsets = 0
            self._episode_total_onsets = 0
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._episode_gt_onsets = []
        self._episode_robot_onsets = []
        self._episode_onset_trace = []
        self._episode_unmatched_onsets = 0
        self._episode_total_onsets = 0
        return self._environment.reset()

    def get_velocity_metrics(self) -> Dict[str, float]:
        """Returns velocity statistics over the last `deque_size` episodes."""
        trace = [row for ep in self._all_onset_traces for row in ep]
        if not trace:
            return {}
        if not any(row["matched"] for row in trace):
            return {}
        return self._compute_velocity_metrics_from_trace(trace)

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

        metrics = {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
            "onset_precision": _mean(self._onset_precisions),
            "onset_recall": _mean(self._onset_recalls),
            "onset_f1": _mean(self._onset_f1s),
            "sustain_precision": _mean(self._sustain_precisions),
            "sustain_recall": _mean(self._sustain_recalls),
            "sustain_f1": _mean(self._sustain_f1s),
            "audio_similarity": _mean(
                [row["audio_similarity"] for row in self._episode_metric_rows]
            ),
            "audio_fidelity": _mean(
                [row["audio_fidelity"] for row in self._episode_metric_rows]
            ),
        }
        if self._success_criteria is not None and self._episode_metric_rows:
            metrics["success_rate"] = _mean(
                [row["success"] for row in self._episode_metric_rows]
            )
        return metrics

    def get_audio_metrics(self) -> Dict[str, float]:
        """Returns aggregated render-based audio metrics."""
        if not self._episode_metric_rows:
            return {}

        def _mean_key(key: str) -> float:
            values = [float(row[key]) for row in self._episode_metric_rows if key in row]
            if not values:
                return float("nan")
            return float(sum(values) / len(values))

        return {
            "audio_wave_mae": _mean_key("audio_wave_mae"),
            "audio_wave_rel_mae": _mean_key("audio_wave_rel_mae"),
            "audio_rms_env_mae": _mean_key("audio_rms_env_mae"),
            "audio_rms_env_rel_mae": _mean_key("audio_rms_env_rel_mae"),
            "audio_rms_env_corr": _mean_key("audio_rms_env_corr"),
        }

    def get_episode_metrics_table(self) -> List[dict]:
        """Returns per-episode evaluation metric rows for the last deque window."""
        return [dict(row) for row in self._episode_metric_rows]

    def get_latest_episode_metrics(self) -> Dict[str, float]:
        """Returns the most recent per-episode metric row."""
        if not self._episode_metric_rows:
            return {}
        return dict(self._episode_metric_rows[-1])

    def get_latest_episode_velocity_trace(self) -> List[dict]:
        """Returns the most recent per-episode onset trace."""
        if not self._all_onset_traces:
            return []
        return [dict(row) for row in self._all_onset_traces[-1]]

    def get_latest_audio_renderings(self) -> Dict[str, object]:
        """Returns the most recent rendered robot/GT audio pair."""
        if not self._latest_audio_renderings:
            return {}
        result: Dict[str, object] = {}
        for key, value in self._latest_audio_renderings.items():
            result[key] = value.copy() if isinstance(value, np.ndarray) else value
        return result

    # Helper methods.

    def _build_episode_metric_row(
        self,
        key_press_metrics: EpisodeMetrics,
        sustain_metrics: EpisodeMetrics,
        onset_metrics: EpisodeMetrics,
    ) -> Dict[str, float]:
        velocity_metrics = self._compute_velocity_metrics_from_trace(
            self._episode_onset_trace
        )
        audio_metrics = self._compute_rendered_audio_metrics(self._environment.task)
        row = {
            "precision": float(key_press_metrics.precision),
            "recall": float(key_press_metrics.recall),
            "f1": float(key_press_metrics.f1),
            "onset_precision": float(onset_metrics.precision),
            "onset_recall": float(onset_metrics.recall),
            "onset_f1": float(onset_metrics.f1),
            "sustain_precision": float(sustain_metrics.precision),
            "sustain_recall": float(sustain_metrics.recall),
            "sustain_f1": float(sustain_metrics.f1),
            "matched_onsets": float(sum(row["matched"] for row in self._episode_onset_trace)),
            "unmatched_onsets": float(self._episode_unmatched_onsets),
            "total_robot_onsets": float(self._episode_total_onsets),
        }
        row.update(velocity_metrics)
        row.update(audio_metrics)
        audio_similarity = float(row.get("audio_similarity", 0.0))
        row["audio_fidelity"] = _harmonic_mean(
            float(onset_metrics.f1), audio_similarity
        )
        if self._success_criteria is not None:
            row["success"] = float(self._is_success(row))
        return row

    def _compute_velocity_metrics_from_trace(self, trace: Sequence[dict]) -> Dict[str, float]:
        if not trace:
            return {}

        matched_rows = [row for row in trace if row["matched"]]
        if not matched_rows:
            return {}

        robot_arr = np.array([row["robot_midi_vel"] for row in matched_rows])
        gt_arr = np.array([row["gt_midi_vel"] for row in matched_rows])
        robot_qvel_arr = np.array(
            [row["robot_qvel"] for row in matched_rows], dtype=np.float64
        )
        errors = robot_arr - gt_arr

        return {
            "mean_robot_midi_vel": float(np.mean(robot_arr)),
            "std_robot_midi_vel": float(np.std(robot_arr)),
            "velocity_mae": float(np.mean(np.abs(errors))),
            "velocity_mse": float(np.mean(errors**2)),
            "velocity_bias": float(np.mean(errors)),
            "max_robot_onset_qvel": float(np.max(robot_qvel_arr)),
            "p90_robot_onset_qvel": float(np.percentile(robot_qvel_arr, 90)),
        }

    def _render_audio_from_events(
        self,
        events: Sequence[midi_message.MidiMessage],
        *,
        target_num_samples: Optional[int] = None,
    ) -> np.ndarray:
        playable = any(
            isinstance(event, (midi_message.NoteOn, midi_message.NoteOff))
            for event in events
        )
        if not events or not playable:
            if target_num_samples is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros((target_num_samples,), dtype=np.float32)

        if self._audio_synth.sustained:
            self._audio_synth.sustain_off()
        self._audio_synth.all_notes_off()
        self._audio_synth.all_sounds_off()

        waveform = self._audio_synth.get_samples(
            _sort_midi_events(events), normalize=False
        )

        if self._audio_synth.sustained:
            self._audio_synth.sustain_off()
        self._audio_synth.all_notes_off()
        self._audio_synth.all_sounds_off()

        waveform_float = waveform.astype(np.float32) / _INT16_MAX
        if target_num_samples is not None and len(waveform_float) < target_num_samples:
            waveform_float = np.pad(
                waveform_float, (0, target_num_samples - len(waveform_float))
            )
        return waveform_float

    def _compute_rendered_audio_metrics(self, task) -> Dict[str, float]:
        gt_events = _score_midi_events_from_trajectory(task)
        robot_events = _robot_midi_events_from_episode(
            key_presses=self._key_presses,
            sustain_presses=self._sustain_presses,
            onset_trace=self._episode_onset_trace,
            control_timestep=float(task.control_timestep),
        )

        gt_waveform = self._render_audio_from_events(gt_events)
        robot_waveform = self._render_audio_from_events(
            robot_events, target_num_samples=len(gt_waveform)
        )
        robot_waveform, gt_waveform = _align_waveforms(robot_waveform, gt_waveform)

        if gt_waveform.size == 0 and robot_waveform.size == 0:
            self._latest_audio_renderings = {
                "sample_rate": self._audio_sample_rate,
                "robot_waveform": np.zeros((0,), dtype=np.float32),
                "gt_waveform": np.zeros((0,), dtype=np.float32),
                "robot_events": [],
                "gt_events": [],
            }
            return {
                "audio_wave_mae": 0.0,
                "audio_rms_env_mae": 0.0,
                "audio_rms_env_corr": 1.0,
                "audio_similarity": 1.0,
            }

        wave_mae = float(np.mean(np.abs(robot_waveform - gt_waveform)) / 2.0)
        gt_wave_scale = float(np.mean(np.abs(gt_waveform)))
        wave_similarity = _similarity_from_relative_mae(wave_mae, gt_wave_scale)
        wave_rel_mae = float(wave_mae / (gt_wave_scale + 1e-8))
        robot_env = _rms_envelope(
            robot_waveform, sample_rate=self._audio_sample_rate
        )
        gt_env = _rms_envelope(gt_waveform, sample_rate=self._audio_sample_rate)
        robot_env, gt_env = _align_waveforms(robot_env, gt_env)
        env_mae = float(np.mean(np.abs(robot_env - gt_env))) if gt_env.size else 0.0
        gt_env_scale = float(np.mean(gt_env)) if gt_env.size else 0.0
        env_similarity = _similarity_from_relative_mae(env_mae, gt_env_scale)
        env_rel_mae = float(env_mae / (gt_env_scale + 1e-8)) if gt_env.size else 0.0
        if robot_env.size > 1 and np.std(robot_env) > 0 and np.std(gt_env) > 0:
            env_corr = float(np.corrcoef(robot_env, gt_env)[0, 1])
        else:
            env_corr = float("nan")
        audio_similarity = _harmonic_mean(wave_similarity, env_similarity)

        self._latest_audio_renderings = {
            "sample_rate": self._audio_sample_rate,
            "robot_waveform": robot_waveform.astype(np.float32, copy=True),
            "gt_waveform": gt_waveform.astype(np.float32, copy=True),
            "robot_events": [_serialize_midi_event(event) for event in robot_events],
            "gt_events": [_serialize_midi_event(event) for event in gt_events],
        }
        return {
            "audio_wave_mae": wave_mae,
            "audio_wave_rel_mae": wave_rel_mae,
            "audio_rms_env_mae": env_mae,
            "audio_rms_env_rel_mae": env_rel_mae,
            "audio_rms_env_corr": env_corr,
            "audio_similarity": audio_similarity,
        }

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

    def _compute_onset_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for true onset events over the episode."""
        if not self._episode_gt_onsets:
            return EpisodeMetrics(1.0, 1.0, 1.0)

        y_true = np.concatenate(self._episode_gt_onsets).astype(np.int32)
        y_pred = np.concatenate(self._episode_robot_onsets).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
        )
        return EpisodeMetrics(float(precision), float(recall), float(f1))

    def _is_success(self, row: Dict[str, float]) -> bool:
        assert self._success_criteria is not None
        for metric_name, threshold in self._success_criteria.items():
            if float(row.get(metric_name, float("-inf"))) < float(threshold):
                return False
        return True

    def __del__(self) -> None:
        try:
            self._audio_synth.stop()
        except Exception:
            pass


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(2.0 / (1.0 / a + 1.0 / b))


def _similarity_from_relative_mae(mae: float, reference_scale: float) -> float:
    if reference_scale <= 0.0:
        return 1.0 if mae <= 0.0 else 0.0
    relative_mae = float(mae / (reference_scale + 1e-8))
    return float(np.clip(1.0 - relative_mae, 0.0, 1.0))
