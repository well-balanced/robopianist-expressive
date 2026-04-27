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

"""RoboPianist suite."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from dm_control import composer
from mujoco_utils import composer_utils

from robopianist import music
from robopianist.music.style_transform import apply_style
from robopianist.suite.tasks import piano_with_shadow_hands

# RoboPianist-repertoire-150.
_BASE_REPERTOIRE_NAME = "RoboPianist-repertoire-150-{}-v0"
REPERTOIRE_150 = [_BASE_REPERTOIRE_NAME.format(name) for name in music.PIG_MIDIS]
_REPERTOIRE_150_DICT = dict(zip(REPERTOIRE_150, music.PIG_MIDIS))

# RoboPianist-etude-12.
_BASE_ETUDE_NAME = "RoboPianist-etude-12-{}-v0"
ETUDE_12 = [_BASE_ETUDE_NAME.format(name) for name in music.ETUDE_MIDIS]
_ETUDE_12_DICT = dict(zip(ETUDE_12, music.ETUDE_MIDIS))

# RoboPianist-debug.
_DEBUG_BASE_NAME = "RoboPianist-debug-{}-v0"
DEBUG = [_DEBUG_BASE_NAME.format(name) for name in music.DEBUG_MIDIS]
_DEBUG_DICT = dict(zip(DEBUG, music.DEBUG_MIDIS))

# All valid environment names.
ALL = REPERTOIRE_150 + ETUDE_12 + DEBUG
_ALL_DICT: Dict[str, Union[Path, str]] = {
    **_REPERTOIRE_150_DICT,
    **_ETUDE_12_DICT,
    **_DEBUG_DICT,
}


def load(
    environment_name: str,
    midi_file: Optional[Path] = None,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
    style_velocity_scale: float = 1.0,
    train_style_velocity_scales: Tuple[float, ...] = (),
    style_velocity_contrast: float = 1.0,
    style_melody_gain: float = 1.0,
    style_dynamic_trend: float = 0.0,
) -> composer.Environment:
    """Loads a RoboPianist environment.

    Args:
        environment_name: Name of the environment to load. Must be of the form
            "RoboPianist-repertoire-150-<name>-v0", where <name> is the name of a
            PIG dataset MIDI file in camel case notation.
        midi_file: Path to a MIDI file to load. If provided, this will override
            `environment_name`.
        seed: Optional random seed.
        stretch: Stretch factor for the MIDI file.
        shift: Shift factor for the MIDI file.
        recompile_physics: Whether to recompile the physics.
        legacy_step: Whether to use the legacy step function.
        task_kwargs: Additional keyword arguments to pass to the task.
        style_velocity_scale: Fixed global velocity multiplier applied to every
            episode. Used for evaluation and single-scale training.
        train_style_velocity_scales: Optional tuple of velocity scales to sample
            from at the beginning of each training episode. When provided, the
            MIDI is kept unstyled here and the task reapplies a sampled scale at
            reset time.
    """
    if midi_file is not None:
        midi = music.load(midi_file, stretch=stretch, shift=shift)
    else:
        if environment_name not in ALL:
            raise ValueError(
                f"Unknown environment {environment_name}. "
                f"Available environments: {ALL}"
            )
        midi = music.load(_ALL_DICT[environment_name], stretch=stretch, shift=shift)

    task_kwargs = dict(task_kwargs or {})
    train_style_velocity_scales = tuple(train_style_velocity_scales)

    if train_style_velocity_scales:
        if "style_velocity_scale_choices" in task_kwargs:
            raise ValueError(
                "Pass mixed training scales via train_style_velocity_scales, not "
                "task_kwargs['style_velocity_scale_choices']."
            )
        if style_velocity_scale != 1.0:
            raise ValueError(
                "style_velocity_scale and train_style_velocity_scales cannot be "
                "used together."
            )
        if (
            style_velocity_contrast != 1.0
            or style_melody_gain != 1.0
            or style_dynamic_trend != 0.0
        ):
            raise ValueError(
                "train_style_velocity_scales currently only supports "
                "velocity_scale randomization."
            )
        task_kwargs["style_velocity_scale_choices"] = train_style_velocity_scales

    # Apply velocity style transforms if any are non-default.
    if not train_style_velocity_scales and (
        style_velocity_scale != 1.0
        or style_velocity_contrast != 1.0
        or style_melody_gain != 1.0
        or style_dynamic_trend != 0.0
    ):
        midi = apply_style(
            midi,
            velocity_scale=style_velocity_scale,
            velocity_contrast=style_velocity_contrast,
            melody_gain=style_melody_gain,
            dynamic_trend=style_dynamic_trend,
        )

    return composer_utils.Environment(
        task=piano_with_shadow_hands.PianoWithShadowHands(midi=midi, **task_kwargs),
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
    )


__all__ = [
    "ALL",
    "DEBUG",
    "ETUDE_12",
    "REPERTOIRE_150",
    "load",
]
