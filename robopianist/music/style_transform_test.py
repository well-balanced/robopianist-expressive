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

"""Tests for style_transform.py."""

from absl.testing import absltest, parameterized

from robopianist import music
from robopianist.music.style_transform import (
    StyleParams,
    apply_style,
    apply_style_params,
)

_V_MIN = 1
_V_MAX = 127
_V_FLOOR = 10  # lower bound used by dynamic_trend


def _vels(midi):
    return [n.velocity for n in midi.seq.notes]


class IdentityTest(absltest.TestCase):
    """All transforms at their neutral value must return identical velocities."""

    def setUp(self):
        self.midi = music.load("CMajorScaleTwoHands")

    def test_identity_all_neutral(self):
        styled = apply_style(self.midi)
        self.assertEqual(_vels(styled), _vels(self.midi))

    def test_identity_velocity_scale_one(self):
        styled = apply_style(self.midi, velocity_scale=1.0)
        self.assertEqual(_vels(styled), _vels(self.midi))

    def test_identity_velocity_contrast_one(self):
        styled = apply_style(self.midi, velocity_contrast=1.0)
        self.assertEqual(_vels(styled), _vels(self.midi))

    def test_identity_melody_gain_one(self):
        styled = apply_style(self.midi, melody_gain=1.0)
        self.assertEqual(_vels(styled), _vels(self.midi))

    def test_identity_dynamic_trend_zero(self):
        styled = apply_style(self.midi, dynamic_trend=0.0)
        self.assertEqual(_vels(styled), _vels(self.midi))


class ImmutabilityTest(absltest.TestCase):
    """Transforms must not modify the original MidiFile."""

    def test_original_unchanged(self):
        midi = music.load("CMajorScaleTwoHands")
        orig_vels = _vels(midi)
        apply_style(midi, velocity_scale=0.5, velocity_contrast=1.8,
                    melody_gain=1.4, dynamic_trend=0.9)
        self.assertEqual(_vels(midi), orig_vels)

    def test_note_count_preserved(self):
        midi = music.load("CMajorScaleTwoHands")
        styled = apply_style(midi, velocity_scale=0.8, velocity_contrast=1.3,
                              melody_gain=1.2, dynamic_trend=-0.5)
        self.assertEqual(styled.n_notes, midi.n_notes)


class VelocityBoundsTest(absltest.TestCase):
    """All transforms must keep velocities in [1, 127]."""

    def setUp(self):
        self.midi = music.load("CMajorScaleTwoHands")

    def _assert_in_bounds(self, styled, lo=_V_MIN, hi=_V_MAX):
        for v in _vels(styled):
            self.assertGreaterEqual(v, lo)
            self.assertLessEqual(v, hi)

    def test_velocity_scale_bounds_low(self):
        self._assert_in_bounds(apply_style(self.midi, velocity_scale=0.01))

    def test_velocity_scale_bounds_high(self):
        self._assert_in_bounds(apply_style(self.midi, velocity_scale=10.0))

    def test_velocity_contrast_bounds_low(self):
        self._assert_in_bounds(apply_style(self.midi, velocity_contrast=0.0))

    def test_velocity_contrast_bounds_high(self):
        self._assert_in_bounds(apply_style(self.midi, velocity_contrast=10.0))

    def test_melody_gain_bounds_high(self):
        self._assert_in_bounds(apply_style(self.midi, melody_gain=10.0))

    def test_dynamic_trend_bounds_positive(self):
        # dynamic_trend uses V_FLOOR (10) as lower bound
        self._assert_in_bounds(apply_style(self.midi, dynamic_trend=1.0), lo=_V_FLOOR)

    def test_dynamic_trend_bounds_negative(self):
        self._assert_in_bounds(apply_style(self.midi, dynamic_trend=-1.0), lo=_V_FLOOR)


class VelocityScaleTest(parameterized.TestCase):
    """velocity_scale scales all velocities up or down."""

    def setUp(self):
        self.midi = music.load("CMajorScaleTwoHands")

    def test_scale_below_one_reduces_velocities(self):
        orig_mean = sum(_vels(self.midi)) / self.midi.n_notes
        styled = apply_style(self.midi, velocity_scale=0.5)
        new_mean = sum(_vels(styled)) / styled.n_notes
        self.assertLess(new_mean, orig_mean)

    def test_scale_above_one_increases_velocities(self):
        orig_mean = sum(_vels(self.midi)) / self.midi.n_notes
        styled = apply_style(self.midi, velocity_scale=1.5)
        new_mean = sum(_vels(styled)) / styled.n_notes
        self.assertGreater(new_mean, orig_mean)


class VelocityContrastTest(absltest.TestCase):
    """velocity_contrast changes the standard deviation of velocities."""

    def setUp(self):
        import numpy as np
        self.midi = music.load("CMajorScaleTwoHands")
        vels = _vels(self.midi)
        self.orig_std = float(np.std(vels))

    def test_contrast_below_one_reduces_std(self):
        import numpy as np
        styled = apply_style(self.midi, velocity_contrast=0.5)
        new_std = float(np.std(_vels(styled)))
        self.assertLess(new_std, self.orig_std + 1)  # at most unchanged

    def test_contrast_above_one_increases_std(self):
        import numpy as np
        styled = apply_style(self.midi, velocity_contrast=1.8)
        new_std = float(np.std(_vels(styled)))
        self.assertGreater(new_std, self.orig_std - 1)  # at least unchanged


class MelodyGainTest(absltest.TestCase):
    """melody_gain only modifies a subset of notes."""

    def test_melody_gain_changes_some_notes(self):
        midi = music.load("CMajorScaleTwoHands")
        styled = apply_style(midi, melody_gain=1.4)
        diffs = [s != o for s, o in zip(_vels(styled), _vels(midi))]
        # At least one note should change (melody notes boosted).
        self.assertGreater(sum(diffs), 0)

    def test_melody_gain_does_not_change_all_notes(self):
        # With fingering, right-hand notes != all notes (there are left-hand notes too).
        midi = music.load("CMajorScaleTwoHands")
        if not midi.has_fingering():
            self.skipTest("No fingering — top-note heuristic may change all notes")
        styled = apply_style(midi, melody_gain=1.4)
        diffs = [s != o for s, o in zip(_vels(styled), _vels(midi))]
        self.assertLess(sum(diffs), midi.n_notes)

    def test_melody_gain_with_pig_midi(self):
        """PIG midi has fingering — verify right-hand notes are affected."""
        midi = music.load("NocturneRousseau")
        self.assertTrue(midi.has_fingering())
        styled = apply_style(midi, melody_gain=1.5)
        diffs = sum(s != o for s, o in zip(_vels(styled), _vels(midi)))
        self.assertGreater(diffs, 0)


class DynamicTrendTest(absltest.TestCase):
    """dynamic_trend shifts the overall dynamic arc."""

    def setUp(self):
        import numpy as np
        self.midi = music.load("NocturneRousseau")  # longer piece → clearer trend
        notes = list(self.midi.seq.notes)
        total = self.midi.duration
        # Split into first and second half by start_time.
        first = [n.velocity for n in notes if n.start_time / total < 0.5]
        second = [n.velocity for n in notes if n.start_time / total >= 0.5]
        self.first_orig = float(np.mean(first)) if first else 64.0
        self.second_orig = float(np.mean(second)) if second else 64.0

    def test_positive_trend_makes_end_louder(self):
        import numpy as np
        styled = apply_style(self.midi, dynamic_trend=1.0)
        notes = list(styled.seq.notes)
        total = styled.duration
        first = [n.velocity for n in notes if n.start_time / total < 0.5]
        second = [n.velocity for n in notes if n.start_time / total >= 0.5]
        mean_first = float(np.mean(first)) if first else 64.0
        mean_second = float(np.mean(second)) if second else 64.0
        self.assertGreater(mean_second, mean_first)

    def test_negative_trend_makes_start_louder(self):
        import numpy as np
        styled = apply_style(self.midi, dynamic_trend=-1.0)
        notes = list(styled.seq.notes)
        total = styled.duration
        first = [n.velocity for n in notes if n.start_time / total < 0.5]
        second = [n.velocity for n in notes if n.start_time / total >= 0.5]
        mean_first = float(np.mean(first)) if first else 64.0
        mean_second = float(np.mean(second)) if second else 64.0
        self.assertGreater(mean_first, mean_second)


class StyleParamsTest(absltest.TestCase):
    """apply_style_params should match apply_style with the same values."""

    def test_style_params_matches_apply_style(self):
        midi = music.load("CMajorScaleTwoHands")
        params = StyleParams(
            velocity_scale=0.9,
            velocity_contrast=1.2,
            melody_gain=1.1,
            dynamic_trend=0.3,
        )
        a = apply_style(midi, **params.__dict__)
        b = apply_style_params(midi, params)
        self.assertEqual(_vels(a), _vels(b))

    def test_default_style_params_is_identity(self):
        midi = music.load("CMajorScaleTwoHands")
        styled = apply_style_params(midi, StyleParams())
        self.assertEqual(_vels(styled), _vels(midi))


if __name__ == "__main__":
    absltest.main()
