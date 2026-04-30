# `experiment` > `main` Contribution Summary

비교 기준:

- base: `main`의 마지막 공통 기준 commit  
  `0d9736c64eba5faafdf214ed7d38d648ffbd5c7f`
- compare target: `experiment` HEAD  
  `fa01ff2f24f33296f514202de2990973d64b4bd4`

요약 수치:

- unique commits: `25`
- changed files: `28`
- diff 규모: `+3801 / -35`

이 문서는 단순히 "어떤 파일이 바뀌었는가"가 아니라, `main` 대비 `experiment` 브랜치가 **연구/시스템 관점에서 어떤 기여를 추가했는지**를 흐름 중심으로 정리한다.

주의:

- `.gitignore`, `AGENTS.md`, `CLAUDE.md`, 실험 노트 문서 같은 운영/메타 변경도 diff에는 포함되지만,
  아래 정리는 **핵심 연구 기여**와 **보조 인프라**를 분리해서 설명한다.

## 1. 가장 큰 기여: "정답 키를 누르는 환경"을 "정답 강약까지 맞추는 환경"으로 확장

`main`의 RoboPianist는 본질적으로 key press / sustain 중심 환경이다.  
`experiment` 브랜치의 가장 큰 기여는 이를 **expressive velocity control** 문제로 확장했다는 점이다.

핵심 흐름은 다음과 같다.

1. 피아노 key joint의 실제 onset 속도(`qvel`)를 추적한다.
2. 그 onset 속도를 MIDI velocity `1..127`로 변환한다.
3. GT MIDI velocity와 비교해 reward를 준다.
4. eval에서도 단순 F1이 아니라 velocity / loudness 지표까지 측정한다.
5. score 자체도 velocity scale/style transform으로 바꿔가며 학습/평가할 수 있게 한다.

즉 이 브랜치는 "맞는 음을 치는가?"에서 끝나지 않고,  
"**맞는 음을 맞는 강약으로 치는가?**"를 환경, reward, metric, 실험 설정 전체에 일관되게 집어넣었다.

관련 파일:

- `robopianist/models/piano/piano.py`
- `robopianist/models/piano/midi_module.py`
- `robopianist/suite/tasks/piano_with_shadow_hands.py`
- `robopianist/wrappers/evaluation.py`
- `robopianist/music/velocity_calibration.py`
- `robopianist/music/style_transform.py`
- `robopianist/suite/__init__.py`

## 2. 피아노 출력 레벨 기여: key joint velocity 기반 MIDI velocity 생성

이 부분이 가장 아래쪽 물리 레벨의 변경이다.

### 무엇이 바뀌었나

`piano.py`는 이제 매 step마다:

- 현재 key joint 속도 `self._key_velocities`
- 새 onset이 발생한 순간의 속도 `self._onset_velocities`

를 추적한다.  
구현은 [piano.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/piano/piano.py:167), [piano.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/piano/piano.py:183) 쪽이다.

그다음 `MidiModule.after_substep(...)`가 이 onset 속도를 받아서 note-on velocity를 만든다.  
구현은 [midi_module.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/piano/midi_module.py:189) 이후다.

### 왜 중요한가

이전에는 piano note-on이 사실상 "눌렸느냐/안 눌렸느냐"에 가까운 이산 이벤트였는데,  
이제는 **얼마나 빠르게 눌렸는가**가 MIDI velocity로 출력된다.

즉 policy가 손을 더 세게/약하게 움직인 결과가 실제 MIDI velocity 차이로 이어진다.

### 구현상 핵심 포인트

- `MAX_KEY_VEL = 8.0`
- `QVEL_MIN = 0.39`

을 single source of truth로 두고, 아래 식으로 mapping한다.

```python
midi_velocity = clip((qvel - QVEL_MIN) / (MAX_KEY_VEL - QVEL_MIN) * 126, 0, 126) + 1
```

구현 위치:

- [midi_module.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/piano/midi_module.py:163)
- [midi_module.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/piano/midi_module.py:207)

### 추가 가치

이 브랜치는 단순히 상수 하나를 넣은 게 아니라, 그 상수가 왜 그런지에 대한 물리적 설명도 코드 주석에 남겼다.

- key tip velocity
- hammer velocity
- lever ratio
- minimum playable onset qvel dead zone

까지 포함해서 mapping의 의미를 문서화했다.  
이건 재현성 측면에서도 기여다.

## 3. reward 레벨 기여: velocity reward 도입과 버전 관리

velocity를 출력만 할 수 있으면 아직 학습은 안 된다.  
`experiment` 브랜치는 여기서 한 단계 더 나아가 **velocity reward**를 task에 넣었다.

### reward 구조 확장

`PianoWithShadowHands._set_rewards()`에서 이제 기본 reward:

- `key_press_reward`
- `sustain_reward`
- `energy_reward`

외에 조건부로:

- `fingering_reward` 또는 `ot_fingering_reward`
- `forearm_reward`
- `velocity_reward`

까지 조립한다.  
구현은 [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:210) 이후다.

### velocity reward의 핵심 설계

현재 코드는 velocity reward를 `v1` / `v2` 두 경로로 유지한다.

- v1: 단순 matched true onset velocity accuracy
- v2: onset accuracy + contour/bias + hold stability penalty

구현:

- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:489)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:499)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:557)

### v1 contribution

v1은 의도적으로 단순하다.

흐름:

1. `activation & ~prev_activation`으로 robot 새 onset 검출
2. 같은 timestep의 **GT true onset**만 lookup
3. unmatched onset은 reward `0`
4. matched onset만 sharp `tolerance()`로 점수화

이 설계는 중요한 의미가 있다.

- 기존처럼 score-active 전체를 흐리게 비교하지 않고
- evaluation semantics와 맞춘 **true onset matching** 기준으로 바꿨다

즉 reward와 evaluation의 의미를 더 가깝게 맞춘 것이다.

### v2 contribution

v2는 단순 절대 오차를 넘어서 dynamics 흐름까지 보려는 실험적 reward다.

추가된 항:

- matched onset 절대 정확도
- previous matched onset 대비 contour
- short-window bias penalty
- piece-level extreme weighting
- hold 중 retrigger penalty
- premature release penalty

즉 "한 음씩 맞췄냐"를 넘어서:

- 강약 흐름이 GT와 비슷한가
- 특정 방향으로 계속 세게/약하게 bias collapse하는가
- hold note articulation이 망가지지 않는가

까지 reward에 넣었다.

이 reward 버전 히스토리는 [version_history.md](/home/cv2/wynn/robopianist-expressive/version_history.md:1)에 한국어로 관리되고 있다.  
이 문서 자체도 `experiment` 브랜치의 중요한 운영 기여다.

## 4. perceptual loudness 기여: raw velocity 대신 "사람이 듣는 크기"에 가까운 축 도입

이 브랜치는 velocity를 숫자 그대로만 다루지 않았다.  
FluidSynth 기반 calibration을 통해 **MIDI velocity -> loudness** lookup을 만들고, eval/보조 reward에서 이를 쓰도록 확장했다.

### 무엇이 추가됐나

- calibration asset: `robopianist/music/velocity_calibration.npz`
- helper class: [velocity_calibration.py](/home/cv2/wynn/robopianist-expressive/robopianist/music/velocity_calibration.py:1)
- calibration/example scripts:
  - `examples/calibrate_velocity.py`
  - `examples/compare_velocity_audio.py`
  - `examples/gen_velocity_audio.py`

### 왜 중요한가

같은 MIDI velocity 차이라도 사람 귀에는 선형적으로 안 들린다.  
이 브랜치는 그 문제를 반영해서:

- linear RMS loudness
- dB-normalized loudness

두 축을 제공한다.

핵심 API:

- `VelocityCalibration.loudness(...)`
- `VelocityCalibration.loudness_db(...)`
- `VelocityCalibration.reward(...)`

구현은 [velocity_calibration.py](/home/cv2/wynn/robopianist-expressive/robopianist/music/velocity_calibration.py:26) 이후다.

### 실제 의미

이 덕분에 최종 평가는 단순 `|robot_vel - gt_vel|`이 아니라:

- loudness MAE
- loudness bias
- loudness correlation
- perceptual dynamics score

까지 볼 수 있게 되었다.  
즉 "velocity 숫자를 맞췄냐"보다 "귀에 비슷하게 들리냐"에 가까운 평가가 가능해졌다.

## 5. score/style 레벨 기여: MIDI velocity distribution 자체를 조절하는 style transform

이 브랜치의 또 다른 큰 기여는 environment가 이제 **같은 곡을 다른 dynamics style로 재생성**할 수 있다는 점이다.

### 추가된 transform

[style_transform.py](/home/cv2/wynn/robopianist-expressive/robopianist/music/style_transform.py:15)에서 네 가지 transform을 제공한다.

- `velocity_scale`
- `velocity_contrast`
- `melody_gain`
- `dynamic_trend`

즉 pitch/timing/fingering은 유지하면서 velocity distribution만 바꿀 수 있다.

### 의미

이건 단순 augmentation이 아니라, expressive control 연구에 직접 필요한 기능이다.

- 같은 piece의 louder / softer 버전 생성
- dynamic range 압축/확장
- melody voice 강조
- crescendo / decrescendo style 부여

를 코드 레벨에서 reproducible하게 만든다.

### suite integration

`suite.load(...)`는 이제 아래 style 인자를 직접 받는다.

- `style_velocity_scale`
- `style_velocity_contrast`
- `style_melody_gain`
- `style_dynamic_trend`

구현은 [suite/__init__.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/__init__.py:51) 이후다.

즉 loader 레벨에서:

1. MIDI를 읽고
2. 필요하면 style transform을 적용하고
3. 그 styled MIDI로 task를 만든다

는 흐름이 완성됐다.

## 6. mixed-scale training 기여: 한 policy가 여러 target velocity scale을 함께 보게 함

이건 논문/실험 설계 관점에서 꽤 중요한 확장이다.

### 무엇이 바뀌었나

`suite.load(...)`와 `PianoWithShadowHands`는 이제:

- 고정 `style_velocity_scale`
- episode마다 샘플링하는 `style_velocity_scale_choices`

두 모드를 모두 지원한다.

관련 구현:

- [suite/__init__.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/__init__.py:60)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:94)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:256)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:270)

### 실제 흐름

mixed-scale training에서는:

1. train env가 raw MIDI를 들고 시작
2. episode 시작 시 scale 하나를 샘플링
3. `apply_style(..., velocity_scale=sampled_scale)` 적용
4. `_reset_trajectory()`를 다시 호출
5. goal / sustain / score velocity map이 그 scale 기준으로 재생성

즉 replay buffer 안에 여러 target velocity distribution transition이 섞이게 된다.

이 기능은 interpolation / OOD style generalization 실험의 기반이다.

## 7. observation 레벨 기여: binary goal을 velocity-scaled goal로 바꾸고 residual용 compact obs 추가

이 브랜치는 reward만 바꾼 게 아니라, policy가 보는 observation도 expressive control에 맞게 확장했다.

### goal observable 변경

기존 binary goal은 "이 key를 눌러라"까지만 말해줬다.  
현재 goal observable은 **pressed key 위치에 `velocity / 127` 값을 넣는다**.

구현:

- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:805)
- [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:888)

중요한 점:

- 내부 reward 계산용 `_goal_state`는 여전히 binary
- agent가 보는 observable `goal`만 velocity-scaled

즉 key correctness와 expressive target을 분리한 설계다.

### velocity lookahead / residual obs

추가로:

- `_velocity_goal_state`
- `fingering_velocity` observable placeholder
- `compact_residual_obs`

가 들어갔다.

특히 `compact_residual_obs`는:

- future velocity lookahead
- piano state
- sustain
- 양손 qpos/qvel

을 묶은 작은 residual 전용 observation이다.  
구현은 [piano_with_shadow_hands.py](/home/cv2/wynn/robopianist-expressive/robopianist/suite/tasks/piano_with_shadow_hands.py:915) 이후다.

이건 이후 `robopianist-rl` 쪽 residual experiments를 받쳐주는 환경 측 기여다.

## 8. evaluation 기여: F1-only에서 velocity/loudness-aware metrics로 확장

`main` 대비 `experiment`의 가장 실용적인 차이 중 하나는 evaluation wrapper다.

### 기존 역할

원래 `MidiEvaluationWrapper`는 대체로:

- precision
- recall
- F1

를 보는 wrapper였다.

### experiment에서 추가된 것

현재 wrapper는 episode마다:

- robot onset trace
- matched / unmatched onset
- robot qvel
- robot MIDI velocity
- GT MIDI velocity
- needed qvel
- qvel gap

을 모두 기록한다.  
구현은 [evaluation.py](/home/cv2/wynn/robopianist-expressive/robopianist/wrappers/evaluation.py:118) 이후다.

그리고 `get_velocity_metrics()`가 다음을 계산한다.

- `velocity_mae`
- `velocity_mse`
- `velocity_bias`
- `max_robot_onset_qvel`
- `p90_robot_onset_qvel`
- `loudness_mae`
- `loudness_bias`
- `loudness_correlation`
- `dynamic_range_ratio`
- `perceptual_dynamics_score`

구현은 [evaluation.py](/home/cv2/wynn/robopianist-expressive/robopianist/wrappers/evaluation.py:211) 이후다.

### 의미

이제 policy를 평가할 때:

- "맞는 key를 눌렀는가"뿐 아니라
- "세게/약하게 얼마나 잘 맞췄는가"
- "강약 분포와 상관관계가 유지되는가"
- "사람이 듣는 loudness 축에서 bias가 있는가"

를 함께 볼 수 있다.

즉 expressive RL 논문에서 필요한 metric layer를 실질적으로 구현한 셈이다.

## 9. 제어/학습 보조 기여: Lagrangian wrapper와 reduced action space

### LagrangianVelocityWrapper

[lagrangian.py](/home/cv2/wynn/robopianist-expressive/robopianist/wrappers/lagrangian.py:15)는  
`F1 >= target` 제약을 유지하면서 velocity reward coefficient를 adaptive하게 줄이는 wrapper다.

핵심 아이디어:

- key_press reward는 유지
- velocity reward만 `lambda`로 damping
- episode 끝날 때 현재 F1을 읽고 `lambda` 업데이트

즉 "정확한 key press를 먼저 안정화하고, velocity reward는 너무 세게 policy를 흔들지 않게 하자"는 보조 장치다.

비록 이 wrapper가 실험의 main path였는지는 별개로,  
`experiment` 브랜치가 고민한 학습 안정화 방향을 잘 보여주는 코드다.

### reduced action space 조정

[shadow_hand.py](/home/cv2/wynn/robopianist-expressive/robopianist/models/hands/shadow_hand.py:73)에서는 reduced action space에서 제거할 DOF와 thumb range 조정이 들어갔다.

변경 내용:

- `A_THJ5`
- `A_THJ1`
- `A_LFJ5`

를 제거

그리고 thumb joint range를 줄였다.

이건 expressive velocity 자체의 직접 기여라기보다는,  
실험 안정성과 controllability를 위해 hand control space를 정리한 변경이다.

## 10. 테스트와 재현성 기여

`experiment` 브랜치는 기능 추가만 한 게 아니라, 관련 테스트도 꽤 많이 붙였다.

추가/확장된 테스트:

- `robopianist/models/piano/midi_module_test.py`
- `robopianist/music/style_transform_test.py`
- `robopianist/suite/suite_test.py`
- `robopianist/suite/tasks/piano_with_shadow_hands_test.py`
- `robopianist/wrappers/evaluation_test.py`

의미:

- qvel -> MIDI velocity mapping
- style transform correctness
- suite loader integration
- task reward/observable 확장
- eval metric 계산

이 단위로 회귀를 막는 장치를 갖췄다.

또한 calibration/diagnostic용 스크립트:

- `scripts/measure_key_vel.py`
- `examples/calibrate_velocity.py`
- `examples/compare_velocity_audio.py`
- `examples/gen_velocity_audio.py`

도 추가되어 있어서, 단순 코드 변경이 아니라 **실험 재현과 해석 도구**까지 같이 들어왔다.

## 11. 문서/실험 운영 기여

연구 내용 자체와 별개로, `experiment` 브랜치에는 실험 운영 관점의 기여도 있다.

### versioning discipline

[version_history.md](/home/cv2/wynn/robopianist-expressive/version_history.md:1)에:

- `v1.0`
- `v1.0.1`
- `v2.0`
- `v2.1`
- `v2.1.1`
- `mixed-scale training v1.0`
- `OOD interpolation eval v1.0`
- `scale complexity 실험 계획 v1.0`

이 정리되어 있다.

이건 "그때그때 reward를 바꿨다" 수준이 아니라,  
reward semantics 변화와 실험 의도를 버전 단위로 관리한 것이다.

### experiment notes

- [20260428_residual_mixedscale_eval.md](/home/cv2/wynn/robopianist-expressive/docs/experiment_notes/20260428_residual_mixedscale_eval.md)

같은 문서가 추가되어 mixed-scale / interpolation eval 결과를 정리하고 있다.

이런 문서는 논문 작성과 실험 복기 때 실제로 큰 도움이 된다.

## 12. commit 기준으로 압축하면 무엇이 핵심이었나

commit 로그를 기능 축으로 압축하면 대략 이렇게 볼 수 있다.

### A. 피아노 출력과 calibration

- `2e8e433`: key joint velocity 기반 MIDI output
- `44acc4a`: FluidSynth velocity-to-loudness calibration
- `1591441`, `5ea7f74`: velocity mapping 상수 재보정

### B. task reward와 expressive objective

- `bd98794`: velocity reward 도입
- `01ae498`: velocity control tuning
- `fea40aa`: key_press_v2 / coefficient 분리
- `0bc1769`: perceptual loudness reward 실험
- `5fb1ba8`, `d7bc5e8`, `f3853b2`, `f45c8f7`: reward versioning 및 v2 hold penalty 정리

### C. style control / mixed-scale

- `36da8e4`: velocity style transforms
- `585112b`: mixed-scale training
- `9d0e477`, `fa01ff2`: mixed-scale / interpolation 결과 문서화

### D. metric / diagnostics / 학습 보조

- `48a703a`: per-component reward logging
- `81a068c`: Lagrangian wrapper
- `2ba6e73`: tolerant onset matching + naive reward 복원
- `2c0decf`: controllable parts 조정

## 13. 한 문장 요약

`main` 대비 `experiment` 브랜치의 본질적 contribution은:

**RoboPianist를 "맞는 음을 누르는 benchmark"에서 "맞는 음을 맞는 강약과 스타일로 연주하는 expressive RL benchmark"로 확장한 것**이다.

그걸 위해 이 브랜치는:

- 피아노 출력 계층에 velocity를 넣고
- task reward에 expressive objective를 넣고
- evaluation을 perceptual loudness-aware로 바꾸고
- style transform / mixed-scale training으로 interpolation 실험을 가능하게 만들고
- 그 전 과정을 테스트/문서/버전 관리까지 포함해 정리했다.

## Appendix. diff에 포함되지만 핵심 연구 기여로 보긴 어려운 항목

아래는 diff에는 포함되지만, 위 핵심 contribution과는 성격이 다르다.

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md`
- 일부 실험 노트 문서

이들은 연구 아이디어 자체의 기여라기보다 운영/협업 보조 변경이다.
