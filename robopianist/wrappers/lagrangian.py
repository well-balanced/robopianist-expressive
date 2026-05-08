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

"""Lagrangian constraint wrapper for velocity reward shaping.

Maintains F1 >= f1_target as a soft constraint by adaptively scaling the
velocity reward coefficient via a Lagrange multiplier lambda (λ).

Design:
  - key_press reward is NEVER scaled down — λ only suppresses velocity reward.
  - λ ∈ [1, ∞): when F1 is sufficient λ → 1 (velocity at full strength);
    when F1 drops below f1_target, λ increases and velocity is dampened.
  - velocity_coef = base_vel_coef / λ,  where base_vel_coef is read from
    task._velocity_reward_coef at construction time.

Update rule (once per episode end):
  λ ← max(1.0, λ + lam_step * (f1_target - current_f1))

  lam_lr controls sensitivity: larger = faster response to F1 changes.
  A value of 0.1 means an F1 gap of 0.10 shifts λ by 0.01 per episode.

Usage:
    raw_env = suite.load(...)
    task = raw_env.task          # save reference before wrapping
    env = MidiEvaluationWrapper(raw_env, deque_size=1)
    env = LagrangianVelocityWrapper(
        env, task,
        f1_target=0.90,
        lam_lr=0.1,
        lam_init=1.0,            # set > 1 (e.g. 5.0) for a warm-up phase
    )
    # env.lambda_           → current λ  (log to wandb)
    # env.velocity_coef     → current effective velocity coefficient
    # env.get_lagrangian_stats() → dict for wandb.log()
"""

import dm_env
from dm_env_wrappers import EnvironmentWrapper

from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands


class LagrangianVelocityWrapper(EnvironmentWrapper):
    """Adaptively scales velocity reward to maintain an F1 constraint.

    Place this wrapper *outside* MidiEvaluationWrapper so that
    get_musical_metrics() is reachable via self._environment.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        task: PianoWithShadowHands,
        f1_target: float = 0.85,
        lam_lr: float = 0.1,
        lam_init: float = 1.0,
    ) -> None:
        """
        Args:
            environment: Wrapped environment (must have MidiEvaluationWrapper inside).
            task:        PianoWithShadowHands instance — obtained before wrapping via
                         `raw_env = suite.load(...); task = raw_env.task`.
            f1_target:   F1 constraint threshold τ (default 0.90).
            lam_lr:      Dual learning rate — per-episode λ update step size (default 0.1).
                         Larger = faster response; smaller = smoother.
            lam_init:    Initial λ. Use 1.0 for no warm-up, or a larger value
                         (e.g. 5.0) to suppress velocity reward until F1 stabilises.
        """
        super().__init__(environment)
        self._task = task
        self._f1_target = f1_target
        self._lam_lr = lam_lr
        self._lam = float(lam_init)
        # Base coefficient is whatever the task was initialised with.
        self._base_vel_coef = task._velocity_reward_coef
        # Apply initial λ immediately.
        self._task._velocity_reward_coef = self._base_vel_coef / self._lam

    # ------------------------------------------------------------------
    # dm_env interface
    # ------------------------------------------------------------------

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        if timestep.last():
            self._update_lambda()
        return timestep

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_lambda(self) -> None:
        """Read episode F1 and update λ + velocity_reward_coef."""
        try:
            f1 = self._environment.get_musical_metrics()["f1"]
        except (AttributeError, KeyError):
            # MidiEvaluationWrapper not in chain — skip silently.
            return
        self._lam = max(1.0, self._lam + self._lam_lr * (self._f1_target - f1))
        self._task._velocity_reward_coef = self._base_vel_coef / self._lam

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def lambda_(self) -> float:
        """Current Lagrange multiplier λ."""
        return self._lam

    @property
    def velocity_coef(self) -> float:
        """Current effective velocity reward coefficient (= base / λ)."""
        return self._task._velocity_reward_coef

    def get_lagrangian_stats(self) -> dict:
        """Returns a dict ready for wandb.log()."""
        return {
            "lambda": self._lam,
            "velocity_coef": self.velocity_coef,
            "f1_target": self._f1_target,
        }
