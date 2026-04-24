# Version History

이 파일은 velocity reward 및 관련 학습 설정의 버전 변경 내역을 기록한다.
앞으로 버전이 바뀔 때마다 이 파일에 한국어로 누적 기록한다.

## 2026-04-24

### v1.0

- velocity reward를 `v1 / v2`로 분리하고, 기본값은 `v1.0`으로 유지했다.
- `PianoWithShadowHands`에서 `use_velocity_reward_v2: bool = False` 플래그를 추가했다.
- `_compute_velocity_reward()`는 내부에서 `v1` 또는 `v2`로 분기하도록 변경했다.
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

### 실행 토글

- 현재 학습 실행 시 이름에는 reward 버전을 `v1.0`, `v2.0`처럼 표시한다.
- `train.py`에서는 `use_velocity_reward_v2: bool = False` 인자를 통해 reward 버전을 선택한다.
- 기본 실행은 `v1.0`, `--use-velocity-reward-v2`를 추가하면 `v2.0`이다.
