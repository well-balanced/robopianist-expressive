# Version History

이 파일은 velocity reward 및 관련 학습 설정의 버전 변경 내역을 기록한다.
앞으로 버전이 바뀔 때마다 이 파일에 한국어로 누적 기록한다.

## 2026-04-24

### v1.0

- velocity reward를 `v1 / v2`로 분리하고, 기본값은 `v1.0`으로 유지했다.
- `PianoWithShadowHands`에서 reward 버전에 따라 velocity reward 경로를 분기할 수 있게 했다.
- 현재 `v1.0`은 다음 규칙으로 동작한다.
  - robot의 새 onset만 본다.
  - non-onset step reward는 `0.0`이다.
  - GT의 true onset과 매칭되지 않는 onset reward는 `0.0`이다.
  - GT true onset과 매칭된 경우만 velocity reward를 계산한다.
  - reward는 raw MIDI velocity 차이에 대해 `tolerance()` 기반 gaussian 보상으로 계산한다.
  - 최종 reward는 `velocity_reward_coef * mean(step_rewards)`이다.
- 목적:
  - 기존 loudness 기반 reward가 너무 완만해서 큰 velocity 오차에도 reward가 거의 유지되던 문제를 줄이기 위해서다.
  - reward semantics를 evaluation의 true onset 기준과 더 가깝게 맞추기 위해서다.

### v1.0.1

- `v1.0.1`은 `v1.0`의 onset-only / true-onset-only semantics는 유지하면서, raw MIDI diff 대신 loudness space에서 오차를 보는 버전이다.
- 계산 흐름:
  - robot 새 onset만 본다.
  - GT true onset과 매칭된 key만 reward를 계산한다.
  - `VelocityCalibration.loudness_db()`로 robot/GT velocity를 loudness 공간으로 보낸 뒤,
    `|loudness(robot) - loudness(gt)|`를 sharp한 `tolerance()` 곡선으로 점수화한다.
- 목적:
  - velocity 차이가 구간마다 다르게 들리는 perceptual 차이를 reward에 반영하기 위해서다.
  - 기존 loudness reward의 의미는 유지하면서, 예전 formulation보다 훨씬 날카로운 gradient를 주기 위해서다.

### v2.0

- `v2.0`은 실험용 reward로 추가했다.
- `--use-velocity-reward-v2`를 켜면 `v2.0`을 사용한다.
- 현재 `v2.0`은 다음 항으로 구성된다.
  - matched true onset에 대한 절대 정확도 항
  - 이전 matched onset 대비 contour 항
  - 최근 matched onset error 평균에 대한 bias penalty
  - piece-level GT onset velocity median 기준 extreme weighting
- 목적:
  - 단순 절대 오차뿐 아니라 강약 흐름과 bias collapse까지 같이 보려는 시도다.
- 관찰된 한계:
  - matched true onset에만 reward가 붙기 때문에 held note stability, retrigger, early release를 직접 벌주지 못한다.
  - onset velocity는 좋아질 수 있지만, articulation/sustain 품질은 별도로 깨질 수 있다.

### v2.1

- `v2.1`은 `v2.0`의 onset-accuracy 구조를 유지하면서 hold-stability penalty를 추가한 버전이다.
- 새로 추가된 penalty는 두 가지다.
  - score상 이미 hold 상태여야 하는 key에서 robot이 새 onset을 만든 경우 penalty
  - score상 아직 active여야 하는 key를 robot이 release한 경우 penalty
- 목적:
  - `v2.0`에서 관찰된 late initial onset, held-note retrigger, premature release dead zone을 막기 위해서다.
  - onset velocity 성능은 유지하면서 articulation/sustain 품질을 함께 올리기 위해서다.
- 현재 고정 penalty 계수:
  - unexpected hold onset: `0.75`
  - premature release: `0.50`

### v2.1.1

- `v2.1.1`은 `v2.1`의 구조는 유지하고, hold penalty scale mismatch만 줄이는 최소 수정 버전이다.
- 변경점:
  - episode 시작 직후 몇 step은 hold penalty를 계산하지 않는다.
  - 목적은 `initial_buffer_time` 때문에 score상 이미 눌려 있어야 하는 음을 시작부에서 늦게 누르는 현상을 unfair penalty로 세게 때리지 않기 위해서다.
  - unexpected hold onset penalty를 `0.75 -> 0.20`으로 낮췄다.
  - premature release penalty를 `0.50 -> 0.10`으로 낮췄다.
- 목적:
  - 예전 `v2 coef=1.0`과 비교할 때, hold penalty가 objective 전체를 뒤집어버리는 문제를 줄이기 위해서다.
  - held-note retrigger / premature release는 여전히 억제하되, onset velocity 학습 자체를 덮어버리지는 않게 하려는 조정이다.

### 실행 토글

- 실행 이름에는 실험 당시의 reward 버전을 그대로 표시한다. 예: `v1.0`, `v1.0.1`, `v2.1`, `v2.1.1`
- 다만 코드 안에는 예전 버전을 계속 분기해서 남기지 않는다.
- 현재 소스는 `현재 v1`과 `현재 v2` 두 경로만 유지한다.
- 예전 세부 버전 구현은 git history와 이 문서로 추적한다.
- 기존 `--use-velocity-reward-v2` 플래그는 backward compatibility용으로 유지된다.
- `velocity_reward_version` 문자열 인자도 launch 호환성 때문에 받을 수는 있지만, 현재 코드는 `v2*` 문자열이면 최신 v2 경로를, 그 외는 최신 v1 경로를 사용한다.

### 기타 관련 변경

- `robopianist/music/velocity_calibration.py`의 loudness helper reward는 `-(delta * delta)`에서 `-abs(delta)`로 변경되었다.
- 이 변경은 loudness 차이에 대한 penalty 기울기를 더 크게 만들기 위한 조정이다.
- 다만 현재 `v1.0`, `v2.0`의 main reward 경로는 `tolerance()` 기반 보상을 직접 사용하므로, 이 helper 변경은 현재 active reward의 핵심 계산 경로와는 분리되어 있다.

## 2026-04-26

### mixed-scale training v1.0

- 한 policy가 여러 target velocity distribution을 같이 보도록, per-episode velocity scale sampling 경로를 추가했다.
- 새 학습 인자:
  - `suite.load(..., train_style_velocity_scales=(0.8, 1.0, 1.2))`
  - `train.py --train-style-velocity-scales 0.8 1.0 1.2`
- 동작 흐름:
  - train env는 raw MIDI를 task로 넘긴다.
  - `PianoWithShadowHands.initialize_episode()`에서 episode 시작마다 scale 하나를 샘플링한다.
  - 샘플링된 scale로 `apply_style(..., velocity_scale=sampled_scale)`를 다시 적용한다.
  - 그 뒤 `_reset_trajectory()`를 다시 호출해서 score note / sustain / velocity target map을 현재 scale 기준으로 재생성한다.
- 목적:
  - `0.8 / 1.0 / 1.2`를 모두 본 단일 policy를 학습시켜서, `0.9 / 1.1` 같은 interpolation scale generalization을 평가하기 위해서다.
  - 기존의 순차 fine-tuning (`1.0 -> 0.8 -> 1.2`) 대신, 한 replay buffer 안에 여러 scale transition을 섞어 넣는 학습을 지원하기 위해서다.
- train / eval 분리:
  - train env만 `train_style_velocity_scales`를 사용한다.
  - eval env는 기존처럼 고정 `style_velocity_scale` 하나만 사용한다.
  - 따라서 학습은 mixed-scale로 하고, 평가는 `0.8 / 0.9 / 1.0 / 1.1 / 1.2`를 각각 따로 돌릴 수 있다.
- 현재 제한:
  - mixed-scale path는 지금 `velocity_scale` randomization만 지원한다.
  - `velocity_contrast`, `melody_gain`, `dynamic_trend`와의 동시 mixed sampling은 아직 지원하지 않는다.

## 2026-04-28

### OOD interpolation eval v1.0

- mixed-scale training (`{0.8, 1.0, 1.2}`)으로 학습한 policy (`04271311_tw_residual_full_a0.1_v1.0_coef0.5_allscale_from2M_2M`)에 대해 dense interpolation eval을 수행했다.
- 평가 설정:
  - scale 범위: `0.70 ~ 1.30`, 간격 `0.05`, 총 13개 scale
  - 에피소드 수: scale당 100 episodes
  - 총 environment steps: 약 205,400 steps (episode length 158 × 1,300 episodes)
  - 소요 시간: 약 44분 (GPU 1개, RTX 5090)
  - 결과 저장: `robopianist-rl/ood_interpolation_results.csv`, wandb group `ood-interpolation-dense-v1`
- 주요 평가 지표:
  - `eval/return`, `eval/f1`, `eval/precision`, `eval/recall`
  - `eval/velocity_mae`, `eval/velocity_bias`
  - `eval/loudness_correlation`, `eval/perceptual_dynamics_score`
  - `eval/mean_robot_midi_vel`
- 결과 요약:
  - **scale 0.70**: `velocity_mae=11.1`, `loudness_correlation=0.27`, `perceptual_dynamics_score=0.24` — 성능이 크게 떨어짐. 학습 범위 밖의 OOD 구간임이 명확함.
  - **scale 0.75**: `velocity_mae=3.8`, `loudness_correlation=0.76`, `perceptual_dynamics_score=0.86` — 급격히 회복. 경계 구간.
  - **scale 0.80 ~ 1.20** (학습 범위 내): 전반적으로 안정적. `velocity_mae` 2.5~5.4 수준.
  - **scale 0.90 ~ 1.00**: `velocity_mae` 가장 낮고 `loudness_correlation` 0.90 이상 — 가장 성능이 좋은 구간.
  - **scale 1.05 ~ 1.20**: `perceptual_dynamics_score`가 학습 범위 내임에도 0.58~0.80으로 변동. 특히 `1.05, 1.10`에서 `loudness_correlation`이 하락하는 경향.
  - **scale 1.25 ~ 1.30**: `velocity_mae` 6.2~10.2, `perceptual_dynamics_score` 0.26~0.52 — 다시 급락. OOD 구간.
- 결론:
  - 학습 scale `{0.8, 1.0, 1.2}` 기준으로 **0.75~1.20** 구간은 어느 정도 generalization이 된다.
  - **0.90~1.10**이 interpolation이 가장 잘 되는 구간이며, OOD claim의 경계로 제시할 수 있다.
  - **0.70 이하, 1.25 이상**은 명확한 OOD 성능 저하가 관찰됨.
- 향후 과제:
  - 왜 굳이 sparse하게 학습하고 interpolation에 의존하느냐는 리뷰어 질문에 대한 답변 확보 필요.
  - 다음 실험: train scale 개수를 늘렸을 때 학습 자체가 힘들어지는지 정량적으로 확인.

## 2026-04-29

### scale complexity 실험 계획 v1.0

- **목적**: "왜 모든 scale을 다 샘플링하지 않고 sparse하게 훈련하느냐"는 리뷰어 질문에 대한 실험적 근거를 확보한다.
- **가설**: train에 사용하는 velocity scale 조합의 수가 많아질수록 학습이 불안정해지고, 수렴이 느려지며, seen scale에서의 성능도 저하된다.
- **실험 매트릭스**:

  | 조건 | train scales | seed 수 | 총 runs |
  |------|-------------|---------|--------|
  | scale1 | `{1.0}` | 3 | 3 |
  | scale2 | `{0.8, 1.2}` | 3 | 3 |
  | scale3 | `{0.8, 1.0, 1.2}` | 3 | 3 |
  | scale5 | `{0.8, 0.9, 1.0, 1.1, 1.2}` | 3 | 3 |
  | scale13 | `{0.70, 0.75, ..., 1.30}` (0.05 간격, 13개) | 3 | 3 |

  - 총 15 runs, 서버 2대 (GPU 8개) 기준 2배치로 완료 예상.
  - 각 run: 동일한 base checkpoint (`04241633_tw_v1.0_coef0.5_2M/checkpoint_2000000.flax`), 동일한 총 steps (2M), 동일한 residual 설정 (`fingers_only, alpha=0.1`).
  - seed: `0, 1, 2` 사용. 기존 `04271311` (scale3, seed=42)은 별도 참조용이며, 이 실험에서는 seed 0/1/2로 새로 돌린다.

- **측정 지표**:
  - 학습 안정성: training return curve의 분산 및 수렴 속도 (wandb training curve).
  - seen scale 성능: 각 조건에서 train에 포함된 scale들의 eval return / F1 / velocity_mae 평균.
  - unseen scale generalization: 각 조건 학습 후 `0.9 / 1.1` eval 성능 (별도 OOD eval 스크립트 재사용).

- **예상 결과**:
  - scale 수 증가 → training return 분산 증가, 수렴 속도 저하, seen scale 평균 성능 하락.
  - 이를 통해 "현실적으로 sparse training이 필요하다"는 주장을 실험으로 뒷받침한다.

- **리스크**:
  - scale 수가 많아져도 학습이 잘 되면 당위성이 약해짐. 이 경우 기여점을 "더 적은 훈련 조합으로 더 넓은 generalization"으로 재정의해야 할 수 있음.

- **실험 스크립트 위치**: `robopianist-rl/run_scale_complexity_exp.sh` (작성 예정).
