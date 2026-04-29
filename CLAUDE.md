# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Rules
- you MUST calcel or run or re-run after my approve
- robopianist-expressive/version_history.md랑 robopianist-rl/version_history.md 항상 확인할 것. 코드 베이스에 큰 변화가 있다면 여기에 버저닝하고 업데이트 할 것

## Versioning

- velocity reward나 관련 학습 로직 버전이 바뀌면 반드시 `version_history.md`에 반영할 것.
- 구조나 계산 흐름 자체가 바뀌는 변경은 `v1.0`, `v2.0`처럼 버저닝할 것.
- 같은 로직을 유지한 채 `bounds`, `margin`, `value_at_margin`, 계수, threshold 같은 파라미터만 바뀌는 경우는 `v1.0.1`, `v1.0.2`처럼 patch 버전으로 관리할 것.
- 실행 이름에도 현재 사용한 reward 버전을 명시할 것. 예: `..._v1.0_...`, `..._v1.0.1_...`, `..._v2.0_...`
- 해당 변경이 특정 버전에 대응한다면 커밋 메시지에도 그 버전을 참고할 수 있게 포함할 것. 예: `velocity reward v1.0.1 margin 조정`
- 버전이 바뀌었는데 `version_history.md`와 실행 이름에 반영되지 않은 상태로 실험을 돌리지 말 것.


## ⚠️ CRITICAL: Always Reference the `clean` Branch First

**Before implementing ANYTHING** — new features, flags, observables, wrappers, training scripts, experiment scripts — you MUST first check the `clean` branch of `robopianist-rl`:

```bash
git -C /home/cv2/wynn/robopianist-rl show clean:<file> 2>/dev/null
git -C /home/cv2/wynn/robopianist-rl show clean --stat
```

The `clean` branch (`dc0fb4b`) is a comprehensive snapshot of all prior work including: residual SAC, observation wrappers, distillation, eval scripts, and experiment shell scripts. It contains prior implementations that **must be reused or adapted** rather than reimplemented from scratch.

**If you implement something without checking `clean` first and get it wrong, that is your fault.** The user should never have to correct observable names, wrapper patterns, or flag names that already exist in `clean`.

Workflow:
1. Check `clean` branch for existing implementation
2. Adapt from `clean` to current codebase
3. Only implement from scratch if genuinely absent from `clean`

## Commands

```bash
# Run all tests (parallel)
make test
# or: pytest -n auto

# Run a single test file
pytest robopianist/suite/tasks/piano_with_shadow_hands_test.py

# Format, lint, and type-check
make format
# or individually: black . && ruff --fix . && mypy .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Project Goal

Train an RL agent to play piano **expressively** — not just pressing the right keys at the right time, but also with the correct MIDI velocity, so that the robot's performance sounds perceptually similar to simply playing back the original MIDI file. Velocity support has been added to the environment; the remaining work is ensuring the agent learns to use it well.

## Architecture

RoboPianist is a deep RL benchmark where a pair of simulated Shadow Hand robots must play piano pieces. Built on **MuJoCo** + **dm_control**'s `composer` framework.

### Key layers

**Physics entities** (`robopianist/models/`):
- `piano/` — 88-key piano model with `MidiModule` that tracks key activations and emits MIDI messages. `MAX_KEY_VEL = 3.5 rad/s` is the single source of truth for velocity scaling (maps joint velocity → MIDI velocity 1–127).
- `hands/` — Shadow Hand models (left & right); constants in `shadow_hand_constants.py`.
- `arenas/` — Stage/arena for the scene.

**RL task** (`robopianist/suite/tasks/`):
- `piano_with_shadow_hands.py` — The main task (`PianoWithShadowHands`). Composes the piano + two hands + stage. Reward terms are registered into a `CompositeReward` in `_set_rewards()`. Active reward terms: `key_press`, `sustain`, `energy` (always), plus optional `fingering_reward` or `ot_fingering_reward` (OT-based, used when no fingering annotations exist), `forearm_reward`, and `velocity_reward`.
- `base.py` — `PianoOnlyTask` and `PianoTask` base classes.

**Environment loading** (`robopianist/suite/__init__.py`):
- `robopianist.suite.load(environment_name, ...)` is the main entry point. Named environments map to MIDI files in three datasets: `DEBUG`, `ETUDE_12`, `REPERTOIRE_150`.
- Custom MIDI: pass `midi_file=Path(...)` to override the named environment.

**Music** (`robopianist/music/`):
- `MidiFile` wraps a MIDI file; `NoteTrajectory.from_midi()` discretizes it to control timesteps.
- PIG dataset (150 pieces with fingering annotations) lives in `music/data/pig_single_finger/`.

**Wrappers** (`robopianist/wrappers/`):
- `MidiEvaluationWrapper` — wraps an env to compute precision/recall/F1 on key presses and velocity accuracy across episodes.
- `SoundWrapper` — synthesizes audio via FluidSynth during rollouts.

### Timing
- Physics timestep: 0.005 s (200 Hz); control timestep: 0.05 s (20 Hz, 10× slower).

### Velocity extension (this fork)
This repo extends the upstream with expressive velocity:
- `MidiModule.after_substep` accepts `key_velocities` (joint angular velocities) and converts them to MIDI velocities using `MAX_KEY_VEL = 3.5 rad/s` as a **normalizer** (not a calibration constant — changing it just rescales the output range).
- `PianoWithShadowHands` has a `velocity_reward` term passed via `velocity_reward_coef` (default 1.0 at runtime via `--velocity_coefficient`); disable with `disable_velocity_reward=True`.
- `MidiEvaluationWrapper` tracks per-onset velocity accuracy; `get_episode_velocity_trace()` returns a wandb.Table-ready list logged every 50k steps under `logs/velocity_trace`.

### Velocity reward design notes (from 100만-step experiments)

**Current formulation** (`_compute_velocity_reward`):
- Returns `coef * (1 - deviation/127)` at onset timesteps; returns constant `coef` (= 1.0) on non-onset steps.
- Linear penalty is too gentle: error=60 still gives reward=0.53. The agent gets weak gradient for large deviations.
- Onset steps are sparse (~35 onsets over ~150 steps), non-onset steps return a constant with no gradient.

**Observed failure mode**: Keys with low GT velocity (e.g., key 39, GT=17~29) are consistently over-hit (robot qvel ~1.5–2.1 rad/s, MIDI vel ~55–80, error=40–60). Root cause: pressing softly risks not activating the key, so the agent rationally trades velocity accuracy for a guaranteed key_press reward. The physical minimum qvel to reliably activate a key is higher than what soft notes require.

**Current formulation** (after fixes): `tolerance()` + gaussian sigmoid with bounds=GT±5, margin=40, value_at_margin=0.1. Reward fires on held keys every timestep (not just onset), using `_onset_velocities` vs `_key_onset_gt_vel`.

### Hand observables — canonical names

**ALWAYS check `robopianist/models/hands/base.py` before referencing observable names.** Do NOT guess or invent observable names.

Observable names available on each hand (defined in `base.py` / `shadow_hand.py`):
- `joints_pos` — joint positions (qpos) — **enabled by default**
- `joints_vel` — joint velocities (qvel) — defined in `base.py:94`, enabled via `enable_joints_vel_obs=True`
- `actuators_velocity` — actuator velocity sensor readings (different from `joints_vel`) — defined in `shadow_hand.py`
- `fingertip_positions` — fingertip 3D positions in world coords
- `fingertip_velocity` — fingertip linear velocities in world coords
- `fingertip_force` — fingertip touch sensor readings
- `actuators_force`, `actuators_power` — actuator force/power sensors

`joints_vel` (qvel) is the correct observable for joint angular velocity observations, not `actuators_velocity`. These are distinct. When the user refers to "joint velocity observations", use `joints_vel`.

## Working with the User

The user is not deeply familiar with NumPy/ML library idioms or robopianist internals. When explaining or modifying code, proactively explain patterns like boolean array operations, numpy indexing, and robopianist-specific conventions (e.g. `activation & ~prev_activation`, `np.flatnonzero`, `_onset_velocities` vs `_key_velocities`, `_notes[t]` structure) rather than assuming they are understood.

## Code Patterns Reference

Common numpy and robopianist patterns that appear throughout the task code:

### NumPy idioms

```python
np.flatnonzero(arr)          # indices where arr is non-zero (or True)
                             # e.g. np.flatnonzero([0,1,0,1]) → [1, 3]

arr & ~other                 # boolean AND NOT — "arr is True AND other is False"
                             # used everywhere for onset/release detection

np.clip(val, 0, 126) + 1    # clamp val to [0,126] then shift to [1,127]
                             # standard MIDI velocity conversion pattern

np.full(n, -1, dtype=float) # array of length n filled with -1 (sentinel = "unset")
```

### Piano activation patterns

```python
# Core boolean arrays (88 keys each)
piano.activation             # True = key is currently pressed (past threshold)
self._prev_activation        # activation from the previous timestep (saved in before_step)

# Detecting transitions
new_onsets = activation & ~self._prev_activation   # keys just pressed this step
releases   = ~activation & self._prev_activation   # keys just released this step

# Getting indices from boolean arrays
np.flatnonzero(new_onsets)   # list of key indices that just became active
```

### Note trajectory

```python
self._notes[t]               # list of PianoNote objects at timestep t
                             # each note has: .key (0–87), .velocity (1–127), .fingering (0–9)
self._t_idx                  # current step index into self._notes
self._notes[self._t_idx - 1] # notes for the step that just happened (used in get_reward)

# Building a GT velocity lookup for a timestep
gt_velocity_map = {note.key: note.velocity for note in self._notes[t]}
gt_vel = gt_velocity_map.get(int(key), -1)  # -1 if robot pressed a key not in GT
```

### Velocity tracking

```python
piano._onset_velocities[key] # qvel (rad/s) at the moment key was last pressed — fixed until next press
piano._key_velocities[key]   # current qvel of key — updated every substep (live value)
# Converting qvel → MIDI velocity (1–127)
robot_midi_vel = int(np.clip(qvel / MAX_KEY_VEL * 126, 0, 126)) + 1
```

### Reward building block

```python
from dm_control.utils.rewards import tolerance

# tolerance() returns 1.0 inside bounds, decays to value_at_margin at margin distance
tolerance(
    x,
    bounds=(lo, hi),       # full reward inside [lo, hi]
    margin=40,             # distance beyond bounds where reward = value_at_margin
    sigmoid="gaussian",    # shape of decay
    value_at_margin=0.1,   # reward value at the edge of margin
)

