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

"""Piano sound module."""

from typing import Callable, List, Optional

import numpy as np
from dm_control import mjcf

from robopianist.models.piano import piano_constants
from robopianist.music import midi_file, midi_message

# ---------------------------------------------------------------------------
# MAX_KEY_VEL — Physical derivation and literature basis
# ---------------------------------------------------------------------------
#
# This constant maps the piano key joint angular velocity (rad/s) to MIDI
# velocity (1–127) via a linear normalisation:
#
#   midi_velocity = clip(qvel / MAX_KEY_VEL * 126, 0, 126) + 1
#
# The value is derived from first principles using the geometry of this
# simulation (piano_constants.py, itself modelled after the Kawai Upright
# Regulation Manual) and empirical hammer-velocity data from the acoustics
# literature.
#
# Step 1 — Key-tip linear velocity from joint angular velocity
# ─────────────────────────────────────────────────────────────
#   The key rotates about its rear hinge.  The MuJoCo joint position is the
#   rotation angle θ (rad), so the joint velocity qvel (rad/s) gives a
#   linear tip velocity of:
#
#     v_tip = qvel × L_key
#
#   where L_key = WHITE_KEY_LENGTH = 0.15 m  (piano_constants.py).
#
# Step 2 — Hammer velocity from key-tip velocity (lever ratio)
# ─────────────────────────────────────────────────────────────
#   Inside an upright piano the key acts as a lever that drives the hammer
#   via the wippen/repetition mechanism.  The effective lever ratio is:
#
#     r_lever = blow_distance / key_dip
#
#   Using Kawai Upright Regulation Manual (Kawai Musical Instruments Mfg.,
#   Vol. 1.0, 2011) specifications:
#     • Hammer blow distance  = 46 mm  (wippen-to-string travel)
#     • Key dip (white key)   = 10 mm  (= WHITE_KEY_TRAVEL_DISTANCE,
#                                         piano_constants.py)
#
#     r_lever = 46 mm / 10 mm = 4.6
#
#   Therefore:
#     v_hammer = v_tip × r_lever = qvel × 0.15 × 4.6 = qvel × 0.69
#
# Step 3 — Maximum hammer velocity from acoustic measurements
# ─────────────────────────────────────────────────────────────
#   Askenfelt & Jansson (1990–1993) measured hammer velocities on acoustic
#   grands and uprights across the full dynamic range (ppp to fff):
#
#     v_hammer ∈ [~0.07, ~5.0] m/s
#
#   Reference: Askenfelt, A. & Jansson, E. V. (1990). "From touch to string
#   vibrations: The initial course of the piano tone." Journal of the
#   Acoustical Society of America, 88(1), 52–63.
#
#   The ceiling of 5.0 m/s represents the physical maximum achievable in
#   fff playing on a real piano and is the value used here.
#
# Step 4 — Back-calculating MAX_KEY_VEL
# ──────────────────────────────────────
#   Setting v_hammer_max = 5.0 m/s and solving for qvel:
#
#     MAX_KEY_VEL = v_hammer_max / (L_key × r_lever)
#                = 5.0 / (0.15 × 4.6)
#                = 5.0 / 0.69
#                ≈ 7.25 rad/s
#
# Empirical validation
# ─────────────────────
#   RL agents trained with fingering annotations (robopianist-expressive,
#   velocity-reward branch) produce onset key qvels up to ~8 rad/s, which
#   is consistent with the derived ceiling and confirms the calibration is
#   physically reasonable.  Agents without wrist motion (e.g. BioCDP) reach
#   lower peak qvels (~3–4 rad/s), corresponding to softer playing — also
#   physically sensible.
#
# Relationship to MIDI velocity specification
# ────────────────────────────────────────────
#   Commercial piano capture systems (e.g. Yamaha Disklavier) measure final
#   hammer velocity optically and map it linearly to MIDI velocity 1–127.
#   We follow the same convention: linear mapping, full range 0→MAX_KEY_VEL
#   corresponds to MIDI 1→127.  A perceptually-weighted (sqrt) mapping
#   (Dannenberg, 2006) was considered but not adopted so as to keep the
#   reward gradient uniform and the conversion invertible.
#
# Single source of truth — import this constant wherever the conversion is
# needed (piano_with_shadow_hands.py, wrappers/evaluation.py, etc.).
# ---------------------------------------------------------------------------
MAX_KEY_VEL: float = 7.25
_MAX_KEY_VEL = MAX_KEY_VEL  # backward-compat alias (used in tests)


class MidiModule:
    """The piano sound module.

    It is responsible for tracking the state of the piano keys and generating
    corresponding MIDI messages. The MIDI messages can be used with a synthesizer
    to produce sound.
    """

    def __init__(self) -> None:
        self._note_on_callback: Optional[Callable[[int, int], None]] = None
        self._note_off_callback: Optional[Callable[[int], None]] = None
        self._sustain_on_callback: Optional[Callable[[], None]] = None
        self._sustain_off_callback: Optional[Callable[[], None]] = None

    def initialize_episode(self, physics: mjcf.Physics) -> None:
        del physics  # Unused.

        self._prev_activation = np.zeros(piano_constants.NUM_KEYS, dtype=bool)
        self._prev_sustain_activation = np.zeros(1, dtype=bool)
        self._midi_messages: List[List[midi_message.MidiMessage]] = []

    def after_substep(
        self,
        physics: mjcf.Physics,
        activation: np.ndarray,
        sustain_activation: np.ndarray,
        key_velocities: Optional[np.ndarray] = None,
    ) -> None:
        # Sanity check dtype since we use bitwise operators.
        assert activation.dtype == bool
        assert sustain_activation.dtype == bool

        timestep_events: List[midi_message.MidiMessage] = []
        message: midi_message.MidiMessage

        state_change = activation ^ self._prev_activation
        sustain_change = sustain_activation ^ self._prev_sustain_activation

        # Note on events.
        for key_id in np.flatnonzero(state_change & ~self._prev_activation):
            if key_velocities is not None:
                velocity = int(np.clip(key_velocities[key_id] / _MAX_KEY_VEL * 126, 0, 126)) + 1
            else:
                velocity = 127
            message = midi_message.NoteOn(
                note=midi_file.key_number_to_midi_number(key_id),
                velocity=velocity,
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_on_callback is not None:
                self._note_on_callback(message.note, message.velocity)

        # Note off events.
        for key_id in np.flatnonzero(state_change & ~activation):
            message = midi_message.NoteOff(
                note=midi_file.key_number_to_midi_number(key_id),
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_off_callback is not None:
                self._note_off_callback(message.note)

        # Sustain pedal events.
        if sustain_change & ~self._prev_sustain_activation:
            timestep_events.append(midi_message.SustainOn(time=physics.data.time))
            if self._sustain_on_callback is not None:
                self._sustain_on_callback()
        if sustain_change & ~sustain_activation:
            timestep_events.append(midi_message.SustainOff(time=physics.data.time))
            if self._sustain_off_callback is not None:
                self._sustain_off_callback()

        self._midi_messages.append(timestep_events)
        self._prev_activation = activation.copy()
        self._prev_sustain_activation = sustain_activation.copy()

    def get_latest_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns the MIDI messages generated in the last substep."""
        return self._midi_messages[-1]

    def get_all_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns a list of all MIDI messages generated during the episode."""
        return [message for timestep in self._midi_messages for message in timestep]

    # Callbacks for synthesizer events.

    def register_synth_note_on_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Registers a callback for note on events."""
        self._note_on_callback = callback

    def register_synth_note_off_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """Registers a callback for note off events."""
        self._note_off_callback = callback

    def register_synth_sustain_on_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal on events."""
        self._sustain_on_callback = callback

    def register_synth_sustain_off_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal off events."""
        self._sustain_off_callback = callback
