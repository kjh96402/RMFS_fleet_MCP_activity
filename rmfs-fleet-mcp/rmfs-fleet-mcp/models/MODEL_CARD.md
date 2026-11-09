# Model Card — RMFS 로봇 대수 추정 모델군

- **출처**: Kim, J., Kang, J., & Jung, H. (2026). A simulation-based deep learning approach to operational robot fleet sizing in robotic mobile fulfillment systems. *Computers & Industrial Engineering*, 214, 111910. https://doi.org/10.1016/j.cie.2026.111910
- **문제 정의**: RMFS(Robotic Mobile Fulfillment System)에서 wave 단위로 반복되는 **운영적 로봇 대수 결정** — 보유 로봇 전량 투입 대신 Makespan 최소 로봇 대수를 추정
- **학습 데이터**: 이산사건 시뮬레이션(DES, SimPy) — 11,232 시나리오 × 10회 반복 × 41 fleet 수준 = 4,605,120 runs. 시나리오별 Makespan 최소 로봇 대수를 라벨(y)로 사용
- **입력 Feature(13개 → one-hot 후 21열)**: 수치형 8(주문 수, 스테이션 수, Warehouse dimension, 가용 로봇 수, 전량 투입 Makespan, Robot density, 스테이션당 처리량, Pod당 처리량) + 범주형 5(통로 구성, 스테이션 배치, 주문 할당, SKU-Pod-위치, 로봇 할당 유연성)

## 모델군 (논문 4.2절 설정 · 테스트셋 1,124건 성능, Table 5·6)

| key | 모델 | 설정 | R² | MAE(대) | a20 | Makespan 증가(%) |
|---|---|---|---|---|---|---|
| mlr | Multiple Linear Regression | — | 0.951 | 4.09 | 0.97 | 3.72 |
| svr | Support Vector Regression | RBF, C=1.0, γ=scale | 0.949 | 4.03 | 0.96 | 3.45 |
| rf | Random Forest (배깅) | n=200, depth=None, leaf=3 | 0.948 | 3.69 | 0.97 | 3.70 |
| xgb | XGBoost (부스팅) | n=200, depth=5 | 0.947 | 4.16 | 0.97 | 3.66 |
| mlp | MLP (Base) | 256, BN·ReLU·Dropout 0.3, AdamW | **0.954** | **3.92** | **0.98** | 2.55 |
| cmlp | **C-MLP (최종)** | MLP + 비대칭 손실 α=1.50 | — | 3.96 | — | **2.15** |

`tools/train_models.py` 로 재학습하면 `registry.json` 의 metrics 에 **본인 데이터셋 기준** 테스트 성능이 기록된다 (위 표는 논문 값).

- **운영 효과(예시, 80대 보유 센터, 15 waves)**: wave 평균 10.6대 절감, Makespan 1.4 % 단축 (Table 7)
- **유의사항**: 학습 범위(주문 700~1,300 · 스테이션 4~12 · 로봇 = 스테이션×5~×10) 밖 입력은 `warnings` 로 표시. 추정치는 권고값이며 관리자가 안전 여유(safety buffer)를 더해 최종 결정

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
