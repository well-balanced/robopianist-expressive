# AGENTS.md

This file provides guidance to Codex and other repository-aware coding agents when working with code in this repository.

# Rules
- 실행하거나 종료하거나 re-run 할 때는 반드시 지금 작업하려는 내용과 의도를 말하고 내게 approve 받고 진행할 것
- `wandb` 로그 조회, `rg`, `git diff`, 설정/결과 확인처럼 파일이나 실험 상태를 바꾸지 않는 읽기 전용 분석은 사전 승인 없이 진행해도 됨
- 모든 구현에 대해서는 "반드시" 어떤 것들이 어떤 흐름으로 되었는지 자세하게 설명할 것. 코드랑 함께
- 코드는 항상 human readable하게
- 구조 자체를 일부러 더 복잡하게 만들지마

## Versioning

- velocity reward나 관련 학습 로직 버전이 바뀌면 반드시 `version_history.md`에 한국어로 반영할 것.
- 구조나 계산 흐름 자체가 바뀌는 변경은 `v1.0`, `v2.0`처럼 버저닝할 것.
- 같은 로직을 유지한 채 `bounds`, `margin`, `value_at_margin`, 계수, threshold 같은 파라미터만 바뀌는 경우는 `v1.0.1`, `v1.0.2`처럼 patch 버전으로 관리할 것.
- 실행 이름에도 현재 사용한 reward 버전을 명시할 것. 예: `..._v1.0_...`, `..._v1.0.1_...`, `..._v2.0_...`
- 해당 변경이 특정 버전에 대응한다면 커밋 메시지에도 그 버전을 참고할 수 있게 포함할 것. 예: `velocity reward v1.0.1 margin 조정`
- 버전이 바뀌었는데 `version_history.md`와 실행 이름에 반영되지 않은 상태로 실험을 돌리지 말 것.

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

Train an RL agent to play piano expressively - not just pressing the right keys at the right time, but also with the correct MIDI velocity, so that the robot's performance sounds perceptually similar to simply playing back the original MIDI file. Velocity support has been added to the environment; the remaining work is ensuring the agent learns to use it well.

## Architecture

RoboPianist is a deep RL benchmark where a pair of simulated Shadow Hand robots must play piano pieces. Built on MuJoCo + dm_control's `composer` framework.

### Key layers

**Physics entities** (`robopianist/models/`):
- `piano/` - 88-key piano model with `MidiModule` that tracks key activations and emits MIDI messages. `MAX_KEY_VEL = 3.5 rad/s` is the single source of truth for velocity scaling (maps joint velocity -> MIDI velocity 1-127).
- `hands/` - Shadow Hand models (left & right); constants in `shadow_hand_constants.py`.
- `arenas/` - Stage/arena for the scene.

**RL task** (`robopianist/suite/tasks/`):
- `piano_with_shadow_hands.py` - The main task (`PianoWithShadowHands`). Composes the piano + two hands + stage. Reward terms are registered into a `CompositeReward` in `_set_rewards()`. Active reward terms: `key_press`, `sustain`, `energy` (always), plus optional `fingering_reward` or `ot_fingering_reward` (OT-based, used when no fingering annotations exist), `forearm_reward`, and `velocity_reward`.
- `base.py` - `PianoOnlyTask` and `PianoTask` base classes.

**Environment loading** (`robopianist/suite/__init__.py`):
- `robopianist.suite.load(environment_name, ...)` is the main entry point. Named environments map to MIDI files in three datasets: `DEBUG`, `ETUDE_12`, `REPERTOIRE_150`.
- Custom MIDI: pass `midi_file=Path(...)` to override the named environment.

**Music** (`robopianist/music/`):
- `MidiFile` wraps a MIDI file; `NoteTrajectory.from_midi()` discretizes it to control timesteps.
- PIG dataset (150 pieces with fingering annotations) lives in `music/data/pig_single_finger/`.

**Wrappers** (`robopianist/wrappers/`):
- `MidiEvaluationWrapper` - wraps an env to compute precision/recall/F1 on key presses and velocity accuracy across episodes.
- `SoundWrapper` - synthesizes audio via FluidSynth during rollouts.

### Timing

- Physics timestep: 0.005 s (200 Hz); control timestep: 0.05 s (20 Hz, 10x slower).

### Velocity extension (this fork)

This repo extends the upstream with expressive velocity:
- `MidiModule.after_substep` accepts `key_velocities` (joint angular velocities) and converts them to MIDI velocities using `MAX_KEY_VEL = 3.5 rad/s` as a normalizer (not a calibration constant - changing it just rescales the output range).
- `PianoWithShadowHands` has a `velocity_reward` term passed via `velocity_reward_coef` (default 1.0 at runtime via `--velocity_coefficient`); disable with `disable_velocity_reward=True`.
- `MidiEvaluationWrapper` tracks per-onset velocity accuracy; `get_episode_velocity_trace()` returns a `wandb.Table`-ready list logged every 50k steps under `logs/velocity_trace`.

### Velocity reward design notes (from 100man-step experiments)

**Current formulation** (`_compute_velocity_reward`):
- Returns `coef * (1 - deviation/127)` at onset timesteps; returns constant `coef` (`= 1.0`) on non-onset steps.
- Linear penalty is too gentle: `error=60` still gives reward `= 0.53`. The agent gets weak gradient for large deviations.
- Onset steps are sparse (about 35 onsets over about 150 steps), non-onset steps return a constant with no gradient.

**Observed failure mode**: Keys with low GT velocity (for example, key 39, GT `= 17-29`) are consistently over-hit (robot qvel about `1.5-2.1 rad/s`, MIDI vel about `55-80`, `error=40-60`). Root cause: pressing softly risks not activating the key, so the agent rationally trades velocity accuracy for a guaranteed `key_press` reward. The physical minimum qvel to reliably activate a key is higher than what soft notes require.

**Current formulation** (after fixes): `tolerance()` + gaussian sigmoid with `bounds=GT+-5`, `margin=40`, `value_at_margin=0.1`. Reward fires on held keys every timestep (not just onset), using `_onset_velocities` vs `_key_onset_gt_vel`.

## Working with the User

The user is not deeply familiar with NumPy/ML library idioms or robopianist internals. When explaining or modifying code, proactively explain patterns like boolean array operations, NumPy indexing, and robopianist-specific conventions (for example `activation & ~prev_activation`, `np.flatnonzero`, `_onset_velocities` vs `_key_velocities`, `_notes[t]` structure) rather than assuming they are understood.

## Code Patterns Reference

Common NumPy and robopianist patterns that appear throughout the task code:

### NumPy idioms

```python
np.flatnonzero(arr)          # indices where arr is non-zero (or True)
                             # e.g. np.flatnonzero([0,1,0,1]) -> [1, 3]

arr & ~other                 # boolean AND NOT - "arr is True AND other is False"
                             # used everywhere for onset/release detection

np.clip(val, 0, 126) + 1     # clamp val to [0,126] then shift to [1,127]
                             # standard MIDI velocity conversion pattern

np.full(n, -1, dtype=float)  # array of length n filled with -1 (sentinel = "unset")
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
                             # each note has: .key (0-87), .velocity (1-127), .fingering (0-9)
self._t_idx                  # current step index into self._notes
self._notes[self._t_idx - 1] # notes for the step that just happened (used in get_reward)

# Building a GT velocity lookup for a timestep
gt_velocity_map = {note.key: note.velocity for note in self._notes[t]}
gt_vel = gt_velocity_map.get(int(key), -1)  # -1 if robot pressed a key not in GT
```

### Velocity tracking

```python
piano._onset_velocities[key] # qvel (rad/s) at the moment key was last pressed - fixed until next press
piano._key_velocities[key]   # current qvel of key - updated every substep (live value)

# Converting qvel -> MIDI velocity (1-127)
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
```

## paper list you need to read
- https://arxiv.org/abs/2304.04150
- https://spj.science.org/doi/epdf/10.34133/cbsystems.0104
- https://arxiv.org/html/2407.18178v1
- https://arxiv.org/abs/2503.14545
- https://arxiv.org/html/2408.11048v1
- https://arxiv.org/abs/2511.02504
- https://arxiv.org/html/2503.15481v2 - 이건 sim2real
