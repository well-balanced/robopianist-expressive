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

"""Velocity style transforms for MidiFile objects.

Four composable transforms that reshape MIDI velocity without changing
pitch, timing, or fingering:

  velocity_scale(α)     — global loudness:    v' = α·v
  velocity_contrast(γ)  — dynamic range:      v' = μ + γ(v − μ)
  melody_gain(α_m)      — melody emphasis:    v' = α_m·v  (melody notes only)
  dynamic_trend(k)      — crescendo/decres.:  v' = clip(μ + σ·clip(z + k(2t−1), -2, 2), 10, 127)
                              where z = (v−μ)/(σ+ε), t ∈ [0,1] is normalised time

Usage (all transforms are composable via apply_style):

    from robopianist.music.style_transform import apply_style
    styled = apply_style(
        midi,
        velocity_scale=0.8,
        velocity_contrast=1.3,
        melody_gain=1.4,
        dynamic_trend=-0.5,
    )

Melody detection:
  - If the MidiFile has PIG fingering annotations (has_fingering() == True),
    the right hand (note.part in 1–5) is treated as the melody voice.
    Left hand (note.part in 6–10) is accompaniment.
  - Otherwise a top-note heuristic is used: for each group of notes that
    start within MELODY_TOL seconds of each other, the highest-pitched
    note is considered melody.

note.part convention (NoteSequence protobuf / PIG dataset):
  0         = no fingering assigned
  1–5       = right hand, thumb (1) through pinky (5)
  6–10      = left hand, thumb (6) through pinky (10)
"""

import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from robopianist.music.midi_file import MidiFile

# Tolerance for grouping simultaneous notes in the top-note heuristic (seconds).
_MELODY_TOL: float = 0.010

# MIDI velocity bounds.
_V_MIN: int = 1
_V_MAX: int = 127
_V_FLOOR: int = 10  # lower clip used by dynamic_trend to avoid inaudible notes

# Small constant to avoid division by zero in z-score computation.
_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Melody note detection
# ---------------------------------------------------------------------------


def _right_hand_note_ids(seq) -> set:
    """Returns a set of (start_time_quantised, pitch) pairs for right-hand notes.

    Uses PIG fingering annotation: note.part in 1–5 → right hand.
    """
    ids = set()
    for note in seq.notes:
        if 1 <= note.part <= 5:
            t_q = round(note.start_time / _MELODY_TOL)
            ids.add((t_q, note.pitch))
    return ids


def _top_note_ids(seq) -> set:
    """Returns a set of (start_time_quantised, pitch) pairs for top-voice notes.

    Within each temporal group (notes starting within _MELODY_TOL of each other),
    the highest-pitched note is labelled melody.
    """
    # Collect all pitches per quantised time bucket.
    groups: dict = defaultdict(list)
    for note in seq.notes:
        t_q = round(note.start_time / _MELODY_TOL)
        groups[t_q].append(note.pitch)

    ids = set()
    for t_q, pitches in groups.items():
        ids.add((t_q, max(pitches)))
    return ids


def _melody_ids(midi: MidiFile) -> set:
    """Returns the set of (t_q, pitch) pairs that should receive melody_gain."""
    if midi.has_fingering():
        return _right_hand_note_ids(midi.seq)
    else:
        return _top_note_ids(midi.seq)


# ---------------------------------------------------------------------------
# Velocity statistics helpers
# ---------------------------------------------------------------------------


def _vel_stats(seq):
    """Returns (mean, std) of all note velocities in the sequence."""
    vels = np.array([n.velocity for n in seq.notes], dtype=float)
    return float(vels.mean()), float(vels.std())


# ---------------------------------------------------------------------------
# Individual transforms (operate on a mutable NoteSequence copy)
# ---------------------------------------------------------------------------


def _apply_velocity_scale(seq, alpha: float) -> None:
    """In-place: v' = clip(α·v, V_MIN, V_MAX)."""
    for note in seq.notes:
        note.velocity = int(np.clip(round(alpha * note.velocity), _V_MIN, _V_MAX))


def _apply_velocity_contrast(seq, gamma: float) -> None:
    """In-place: v' = clip(μ + γ(v − μ), V_MIN, V_MAX)."""
    mu, _ = _vel_stats(seq)
    for note in seq.notes:
        v_new = mu + gamma * (note.velocity - mu)
        note.velocity = int(np.clip(round(v_new), _V_MIN, _V_MAX))


def _apply_melody_gain(seq, alpha_m: float, ids: set) -> None:
    """In-place: v' = clip(α_m·v, V_MIN, V_MAX) for melody notes."""
    for note in seq.notes:
        t_q = round(note.start_time / _MELODY_TOL)
        if (t_q, note.pitch) in ids:
            note.velocity = int(
                np.clip(round(alpha_m * note.velocity), _V_MIN, _V_MAX)
            )


def _apply_dynamic_trend(seq, k: float, total_time: float) -> None:
    """In-place dynamic trend transform.

    Formula:
        z      = (v − μ) / (σ + ε)
        t      = note.start_time / total_time   ∈ [0, 1]
        z_new  = clip(z + k·(2t − 1), −2, 2)
        v'     = clip(μ + σ·z_new, V_FLOOR, V_MAX)

    Interpretation: k > 0 → crescendo (soft start, loud end);
                    k < 0 → decrescendo (loud start, soft end).
    The original note-to-note variation is preserved (z-score path) while
    the overall arc is shifted by k·(2t−1).
    """
    if abs(k) < 1e-9:
        return  # no-op
    mu, sigma = _vel_stats(seq)
    for note in seq.notes:
        t = note.start_time / total_time if total_time > 0 else 0.0
        z = (note.velocity - mu) / (sigma + _EPS)
        z_new = float(np.clip(z + k * (2.0 * t - 1.0), -2.0, 2.0))
        v_new = mu + sigma * z_new
        note.velocity = int(np.clip(round(v_new), _V_FLOOR, _V_MAX))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class StyleParams:
    """Container for the four velocity style parameters.

    All parameters default to the identity transform (no change).

    velocity_scale (α):    global loudness multiplier, range [0.5, 1.5]
    velocity_contrast (γ): dynamic range expansion/compression, range [0.5, 1.8]
    melody_gain (α_m):     melody voice emphasis, range [0.7, 1.5]
    dynamic_trend (k):     crescendo (+) / decrescendo (−) arc, range [−1.0, 1.0]
    """

    velocity_scale: float = 1.0
    velocity_contrast: float = 1.0
    melody_gain: float = 1.0
    dynamic_trend: float = 0.0


def apply_style(
    midi: MidiFile,
    velocity_scale: float = 1.0,
    velocity_contrast: float = 1.0,
    melody_gain: float = 1.0,
    dynamic_trend: float = 0.0,
) -> MidiFile:
    """Apply velocity style transforms to a MidiFile and return a new one.

    Transforms are applied in this fixed order:
      1. velocity_scale   — overall loudness
      2. velocity_contrast — dynamic range
      3. melody_gain      — melody voice boost/reduce
      4. dynamic_trend    — crescendo / decrescendo arc

    Args:
        midi:             Source MidiFile (not modified).
        velocity_scale:   α, global velocity multiplier. 1.0 = no change.
        velocity_contrast: γ, dynamic range factor. 1.0 = no change.
        melody_gain:      α_m, melody note velocity multiplier. 1.0 = no change.
        dynamic_trend:    k, crescendo/decrescendo strength. 0.0 = no change.

    Returns:
        A new MidiFile with transformed velocities.
    """
    # deepcopy so we never mutate the frozen original (seq is a protobuf Message,
    # which supports Python deepcopy).
    new_seq = copy.deepcopy(midi.seq)

    total_time = midi.duration  # read from original before any mutation

    # 1. Overall loudness.
    if abs(velocity_scale - 1.0) > 1e-9:
        _apply_velocity_scale(new_seq, velocity_scale)

    # 2. Dynamic range.
    if abs(velocity_contrast - 1.0) > 1e-9:
        _apply_velocity_contrast(new_seq, velocity_contrast)

    # 3. Melody emphasis.  Compute ids from *original* midi so pitch/time are intact.
    if abs(melody_gain - 1.0) > 1e-9:
        ids = _melody_ids(midi)
        _apply_melody_gain(new_seq, melody_gain, ids)

    # 4. Crescendo / decrescendo arc.
    if abs(dynamic_trend) > 1e-9:
        _apply_dynamic_trend(new_seq, dynamic_trend, total_time)

    return MidiFile(seq=new_seq)


def apply_style_params(midi: MidiFile, params: StyleParams) -> MidiFile:
    """Convenience wrapper that accepts a StyleParams dataclass."""
    return apply_style(
        midi,
        velocity_scale=params.velocity_scale,
        velocity_contrast=params.velocity_contrast,
        melody_gain=params.melody_gain,
        dynamic_trend=params.dynamic_trend,
    )
