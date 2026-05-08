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

"""Tests for robopianist.suite."""

import numpy as np
from absl.testing import absltest, parameterized

from robopianist import suite

_SEED = 12345
_NUM_EPISODES = 1
_NUM_STEPS_PER_EPISODE = 10


class RoboPianistSuiteTest(parameterized.TestCase):
    """Tests for all registered tasks in robopianist.suite."""

    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    @parameterized.parameters(*suite.DEBUG)
    def test_task_runs(self, environment_name: str) -> None:
        """Tests task loading and observation spec validity."""
        env = suite.load(environment_name, seed=_SEED)
        random_state = np.random.RandomState(_SEED)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
        self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

        for _ in range(_NUM_EPISODES):
            timestep = env.reset()
            for _ in range(_NUM_STEPS_PER_EPISODE):
                self._validate_observation(timestep.observation, observation_spec)
                if timestep.first():
                    self.assertIsNone(timestep.reward)
                    self.assertIsNone(timestep.discount)
                action = random_state.uniform(
                    action_spec.minimum, action_spec.maximum, size=action_spec.shape
                ).astype(action_spec.dtype)
                timestep = env.step(action)

    def test_train_style_velocity_scales_is_sampled_per_episode(self) -> None:
        env = suite.load(
            environment_name=suite.DEBUG[0],
            seed=_SEED,
            train_style_velocity_scales=(0.8, 1.0, 1.2),
        )

        seen_scales = set()
        for _ in range(8):
            env.reset()
            seen_scales.add(env.task.current_style_velocity_scale)

        self.assertTrue(seen_scales.issubset({0.8, 1.0, 1.2}))
        self.assertGreaterEqual(len(seen_scales), 2)

    def test_train_style_velocity_scales_rejects_fixed_scale_combo(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            suite.load(
                environment_name=suite.DEBUG[0],
                seed=_SEED,
                style_velocity_scale=0.9,
                train_style_velocity_scales=(0.8, 1.0, 1.2),
            )

    def test_train_style_velocity_scales_rejects_other_style_transforms(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports"):
            suite.load(
                environment_name=suite.DEBUG[0],
                seed=_SEED,
                train_style_velocity_scales=(0.8, 1.0, 1.2),
                style_velocity_contrast=1.1,
            )

    def test_train_style_velocity_scales_rejects_task_level_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pass mixed training scales via"):
            suite.load(
                environment_name=suite.DEBUG[0],
                seed=_SEED,
                train_style_velocity_scales=(0.8, 1.0, 1.2),
                task_kwargs={"style_velocity_scale_choices": (0.8, 1.0, 1.2)},
            )


if __name__ == "__main__":
    absltest.main()
