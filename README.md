# RF Post-training 비교 실험

2차원 토이 Rectified Flow 모델에 세 가지 후학습 기법을 적용해, 피드백 종류·비용·분포 보존에 따라 어떤 기법이 맞는지 비교한다.

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_fm_post/blob/main/rf_post_training_colab.ipynb)

## 실행

1. 위 링크로 노트북을 연다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택한다.
3. 위에서 아래로 실행한다. 마지막 셀이 결과 ZIP을 내려받는다.

C만 볼 때는 첫 코드 셀, 설정 셀, 사전학습 셀, C 준비 셀, C 학습 셀만 실행하면 된다.

## 설정

- 데이터: 반지름 2인 원 위의 가우시안 군집 8개, 표준편차 0.12
- 목표: 오른쪽 위 군집. 매끄러운 보상은 폭 0.6의 가우시안, 성공 여부는 반경 0.35
- 모델: 너비 128, 은닉층 3개 MLP. 생성 32단계 Euler. 시간 0은 잡음, 1은 데이터
- 사전학습 2,500회, 후학습 100회, 시드 3개, 업데이트당 샘플 256개, 평가용 고정 잡음 2,048개

모든 값은 `rf_lab/config.py`의 `Config`와 노트북 설정 셀에서 바꾼다.

## 기법

| 기법 | 구현 | 필요한 피드백 |
| --- | --- | --- |
| RL | Flow-GRPO | 점수 |
| Direct | DRaFT 전체 역전파 | 미분 가능한 보상 |
| DPO | Diffusion-DPO (flow-matching 오차) | 선호 쌍 |

**RL (Flow-GRPO)**

- 수집: ODE와 같은 주변분포를 갖는 SDE `dx = [v + σ(t)²/2 · score] dt + σ(t) dW`, `score = (t·v − x)/(1 − t)`
- 잡음: `σ(t) = a·sqrt((1 − t)/t)`, `a = sde_noise = 0.7`. 양 끝 `sde_epsilon = 0.02` 구간은 ODE로 연결
- advantage: 배치 256개를 한 그룹으로 `(R − 평균)/표준편차`
- 손실: 단계별 전이 확률비에 PPO-clip(0.2, 4 epoch) + `β_KL · KL`
- KL: 수집 경로 위 가우시안 전이의 닫힌형 `‖μθ − μref‖² / (2σ²dt)`. `β_KL = regularization["RL"]`

**Direct (DRaFT)**

- 32단계 ODE 전체로 `−평균 보상`을 역전파한다.
- 기울기가 없으면(점수만 제공, 0/1 보상) 업데이트하지 않는다.

**DPO (Diffusion-DPO)**

- 손실: `−log σ(−β·T·[(e_w − e_w,ref) − (e_l − e_l,ref)])`. `e`는 flow-matching 제곱 오차(차원 합), T는 생성 단계 수, ω(λ) = 1
- winner/loser는 같은 t와 noise를 공유한다.
- 선호 쌍은 사전학습 모델에서 뽑은 후보 4,096개로 한 번 만들고, 동점은 제외한다.

## 실험

| 실험 | 보상 | 비교 |
| --- | --- | --- |
| A | 매끄러운 보상 | 세 기법, 기울기 제공·점수만 제공 |
| B | 성공 여부 (0/1) | 기본 반경 0.35·작은 반경 0.08 |
| C | 저장된 선호 기록 | 전체 후보·목표 주변(반경 0.7) 제외 후보 |
| 스윕 | 매끄러운 보상 | RL KL 강도, DPO β, 기법별 학습률 |

## 이전 실행 결과

커밋 `b60e7e7`, Colab T4, 100회 후 시드 3개 평균. 이 커밋의 구현은 지금과 세 군데가 다르다: RL advantage는 평균만 빼고, σ는 상수 0.3이며, DPO logit은 `β × 오차 평균 차이`(지금 식으로 β ≈ 1/32).

| 조건 | 사전학습 | RL | Direct | DPO |
| --- | --- | --- | --- | --- |
| A 기울기 제공 | 10.1% | 97.3% | 62.1% | 14.8% |
| A 점수만 제공 | 10.1% | 97.3% | 10.1% (건너뜀) | 14.8% |
| B 기본 반경 | 10.1% | 97.0% | 10.1% (건너뜀) | 32.6% |
| B 작은 반경 | 1.8% | 90.3% | 1.8% (건너뜀) | 9.2% |
| C 전체 후보 | 10.1% | — | — | 14.8% |
| C 목표 주변 제외 | 10.1% | — | — | 11.9% |

값은 목표 도달률이다. 학습 시간은 RL 약 28초, Direct 약 3초, DPO 1초 미만이다. 학습용 보상 평가 수는 RL·Direct 25,600회, DPO 4,096회다.

- RL은 모든 피드백에서 학습했고, 같은 평가 수에서 보상이 가장 높았다. 시드 0에서 도착점의 약 94%가 목표 군집 하나에 모였다.
- Direct는 학습 시간 대비 가장 빨랐고, 기울기가 없는 조건에서는 움직이지 않았다.
- DPO는 도달률 상승이 작았고, 시드 0의 군집 밖 비율이 A에서 약 48%, B에서 약 62%로 늘었다. 목표 주변 후보를 빼면 상승 폭이 약 1/3로 줄었다.
- C 전체 후보는 A의 DPO와 같은 후보·선호 쌍을 쓰므로 수치가 같다.

## 산출물

`outputs/<실행 시각>/`과 같은 이름의 ZIP에 저장한다.

| 경로 | 내용 |
| --- | --- |
| `figures/` | 벡터장, 변화량, 생성 경로, 도착점 분포, 학습 곡선, 한 업데이트 상세 |
| `figures/<실험>/` | 실험별 그림 |
| `figures/sweeps/` | 스윕별 최종 보상·도달률·군집 밖 비율 |
| `animations/` | 학습 진행·생성 진행 HTML |
| `results/metrics.csv` | 실험·기법·시드·체크포인트별 지표 |
| `results/config.json`, `results/runs.json` | 전체 설정, 실험별 변경 설정, 라이브러리 버전 |
| `results/preferences_*.pt`, `results/one_update_*.pt` | 선호 기록, 한 업데이트의 입력과 신호 |
| `checkpoints/` | 사전학습·기법별 모델, 고정 평가 잡음 |
| `실험결과.md` | 결과 표와 그림 |

학습 시간에는 수집·업데이트·DPO 선호 자료 준비가 들어가고, 사전학습·평가·그림·저장은 빠진다. 군집 밖 비율은 가장 가까운 군집 중심에서 표준편차의 3배보다 먼 점의 비율이다.

## 코드

| 파일 | 역할 |
| --- | --- |
| `rf_lab/config.py` | 설정 |
| `rf_lab/core.py` | 모델, 데이터, 보상, ODE, SDE, 전이 확률·KL, 지표 |
| `rf_lab/training.py` | 사전학습, 세 기법의 업데이트, 선호 쌍 |
| `rf_lab/experiment.py` | 시드별 실험, 스윕, 비용 기록, 저장 |
| `rf_lab/plots.py` | 그림, 애니메이션 |
| `rf_lab/widgets.py` | 노트북 조작부 |
| `rf_lab/export.py` | 보고서, ZIP |
| `rf_lab/checks.py` | 학습 신호·통합 확인 |

실험 설계: [RF_post_training_실험설계.md](RF_post_training_실험설계.md). 수식 참고: [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)의 diffusion post-training 튜토리얼.
