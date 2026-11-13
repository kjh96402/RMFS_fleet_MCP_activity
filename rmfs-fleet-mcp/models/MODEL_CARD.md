# Model Card — RMFS 로봇 대수 추정 모델군

- **출처**: Kim, J., Kang, J., & Jung, H. (2026). A simulation-based deep learning approach to operational robot fleet sizing in robotic mobile fulfillment systems. *Computers & Industrial Engineering*, 214, 111910. https://doi.org/10.1016/j.cie.2026.111910
- **문제 정의**: RMFS(Robotic Mobile Fulfillment System)에서 wave 단위로 반복되는 **운영적 로봇 대수 결정** — 보유 로봇 전량 투입 대신 Makespan 최소 로봇 대수를 추정
- **학습 데이터**: `data/train_data.csv` — 이산사건 시뮬레이션(DES, SimPy) 11,232 시나리오 × 10회 반복 × 41 fleet 수준 = 4,605,120 runs. 시나리오별 Makespan 최소 로봇 대수(`TARGET`)를 라벨로 사용
- **입력 형식**: `data/Data_single_row.csv` 의 12개 열 — `ORDER_NUM, SKU_COUNT_OPTIONS, NUM_AISLES, NUM_CROSS_AISLES, NUM_PS, PLACEMENT_CODE, BATCHING_MODE, OPTION, DEDICATED_MODE, MAX_ROBOTS, MAKESPAN_AT_MAX_ROBOTS, NUM_PODS`
- **모델 Feature(13개 → one-hot 후 21열, 논문 Table 2)**: 입력 12개 열에서 `fleet/features.py` 가 계산
  - 수치형 8: `ORDER_NUM` · `NUM_PS` · `WAREHOUSE_DIM`(= NUM_PODS / NUM_PS) · `MAX_ROBOTS` · `MAKESPAN_AT_MAX_ROBOTS` · `ROBOT_DENSITY`(= MAX_ROBOTS / NUM_PODS) · `REQUIRED_THROUGHPUT_PER_PS_HOUR`(= ORDER_NUM × SKU_COUNT_OPTIONS / (NUM_PS × Makespan)) · `REQUIRED_FULFILL_PER_POD_HOUR`(= ORDER_NUM / (NUM_PODS × Makespan))
  - 범주형 5: `AISLE_CONFIGURATION`(통로 15×15 Square · 12×18 General · 9×27 Elongated) · `PLACEMENT_CODE`(BL·BS·OL·OS) · `BATCHING_MODE`(FCFS·SHARED_CLUSTER) · `OPTION`(1·2) · `DEDICATED_MODE`(dedicated·pooled)

## 모델군 (논문 4.2절 설정 · 테스트셋 1,124건)

| key | 모델 | 설정 | 논문 R² | 논문 MAE(대) | 논문 a20 | 논문 Makespan 증가(%) | 이 저장소 R² / MAE |
|---|---|---|---|---|---|---|---|
| mlr | Multiple Linear Regression | — | 0.951 | 4.09 | 0.97 | 3.72 | 0.952 / 4.11 |
| svr | Support Vector Regression | RBF, C=1.0, γ=scale | 0.949 | 4.03 | 0.96 | 3.45 | 0.949 / 4.03 |
| rf | Random Forest (배깅) | n=200, depth=None, leaf=3 | 0.948 | 4.26 | 0.97 | 3.70 | 0.948 / 4.21 |
| xgb | XGBoost (부스팅) | n=200, depth=5 | 0.947 | 4.16 | 0.97 | 3.66 | 0.945 / 4.23 |
| mlp | MLP (Base) | 256, BN·ReLU·Dropout 0.3, AdamW 1e-3, batch 64, 120 epoch | **0.954** | **3.92** | **0.98** | 2.55 | 0.953 / 3.98 |
| cmlp | **C-MLP (최종)** | MLP + 비대칭 손실 α=1.50 | — | 3.96 | — | **2.15** | 0.951 / 3.90 |

논문 값은 Table 5·6, "이 저장소" 값은 `python tools/train_models.py` 로 `data/train_data.csv` 를 같은 설정(90/10 분할, random_state=42)으로 학습한 결과(`registry.json` 에 기록). 분할·초기값 난수가 달라 소수 둘째 자리에서 차이가 난다. Makespan 증가율은 시뮬레이터가 필요해 이 저장소에서는 계산하지 않는다.

- **운영 효과(예시, 80대 보유 센터, 15 waves)**: wave 평균 10.6대 절감, Makespan 1.4 % 단축 (논문 Table 7)
- **유의사항**: 학습 범위(ORDER_NUM 700~1,300 · NUM_PS 4~12 · MAX_ROBOTS = NUM_PS×10 · NUM_PODS 1,976~2,800 · SKU_COUNT_OPTIONS 2) 밖 입력은 `warnings` 로 표시. 추정치는 권고값이며 관리자가 안전 여유(safety buffer)를 더해 최종 결정

## 파일 구성 (trained 모드)

```
models/
 ├─ registry.json        모델 목록·파일·성능·기본 모델
 ├─ mlr.joblib · svr.joblib · rf.joblib · xgb.joblib    sklearn Pipeline(전처리 포함)
 ├─ preprocess.joblib    z-score + one-hot (MLP 계열 공용)
 ├─ mlp_state.pt · cmlp_state.pt                        torch state_dict (Linear-BN-ReLU-Dropout-Linear)
 └─ MODEL_CARD.md
```
`registry.json` 이 없으면 서버는 **demo 모드**(휴리스틱 1개)로 기동한다.
