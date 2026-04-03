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

"""Tests for midi_module.py."""

from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized

from robopianist.models.piano import midi_module
from robopianist.models.piano import piano_constants as consts
from robopianist.music import midi_message

_NUM_KEYS = consts.NUM_KEYS


def _make_physics(time: float = 0.0):
    physics = mock.MagicMock()
    physics.data.time = time
    return physics


def _make_module() -> midi_module.MidiModule:
    module = midi_module.MidiModule()
    module.initialize_episode(_make_physics())
    return module


def _step(
    module: midi_module.MidiModule,
    activation: np.ndarray,
    key_velocities=None,
    time: float = 0.0,
) -> list:
    sustain = np.zeros(1, dtype=bool)
    module.after_substep(_make_physics(time), activation, sustain, key_velocities)
    return module.get_latest_midi_messages()


def _note_on_velocity(messages: list) -> int:
    note_ons = [m for m in messages if isinstance(m, midi_message.NoteOn)]
    assert len(note_ons) == 1
    return note_ons[0].velocity


class VelocityTest(parameterized.TestCase):
    def test_default_velocity_is_127_without_key_velocities(self) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        messages = _step(module, activation, key_velocities=None)
        self.assertEqual(_note_on_velocity(messages), 127)

    def test_zero_key_velocity_maps_to_midi_velocity_1(self) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        key_velocities = np.zeros(_NUM_KEYS)
        messages = _step(module, activation, key_velocities)
        self.assertEqual(_note_on_velocity(messages), 1)

    def test_max_key_velocity_maps_to_midi_velocity_127(self) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        key_velocities = np.full(_NUM_KEYS, midi_module._MAX_KEY_VEL)
        messages = _step(module, activation, key_velocities)
        self.assertEqual(_note_on_velocity(messages), 127)

    def test_exceeding_max_key_velocity_clamps_to_127(self) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        key_velocities = np.full(_NUM_KEYS, midi_module._MAX_KEY_VEL * 10)
        messages = _step(module, activation, key_velocities)
        self.assertEqual(_note_on_velocity(messages), 127)

    @parameterized.parameters(
        (0.0 * midi_module._MAX_KEY_VEL,),
        (0.25 * midi_module._MAX_KEY_VEL,),
        (0.5 * midi_module._MAX_KEY_VEL,),
        (0.75 * midi_module._MAX_KEY_VEL,),
        (1.0 * midi_module._MAX_KEY_VEL,),
    )
    def test_velocity_increases_monotonically(self, qvel: float) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        key_velocities = np.full(_NUM_KEYS, qvel)
        messages = _step(module, activation, key_velocities)
        midi_vel = _note_on_velocity(messages)
        self.assertBetween(midi_vel, 1, 127)

    def test_faster_press_yields_higher_velocity(self) -> None:
        results = []
        for qvel in [0.5, 2.0, 4.0]:
            module = _make_module()
            activation = np.zeros(_NUM_KEYS, dtype=bool)
            activation[0] = True
            key_velocities = np.full(_NUM_KEYS, qvel)
            messages = _step(module, activation, key_velocities)
            results.append(_note_on_velocity(messages))
        self.assertLess(results[0], results[1])
        self.assertLess(results[1], results[2])

    def test_velocity_only_applied_on_note_on_not_note_off(self) -> None:
        module = _make_module()
        activation = np.zeros(_NUM_KEYS, dtype=bool)
        activation[0] = True
        key_velocities = np.zeros(_NUM_KEYS)
        _step(module, activation, key_velocities, time=0.0)

        # Release key with zero velocity.
        activation[0] = False
        messages = _step(module, activation, key_velocities, time=0.1)
        note_offs = [m for m in messages if isinstance(m, midi_message.NoteOff)]
        self.assertLen(note_offs, 1)


if __name__ == "__main__":
    absltest.main()
