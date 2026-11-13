# rmfs-fleet-mcp — RMFS 로봇 대수 추정 모델을 Python MCP 서버로

Kim, Kang & Jung (2026, *Computers & Industrial Engineering* 214:111910)의 로봇 대수 추정 모델군
(**MLR · SVR · RF · XGB · MLP · C-MLP**)을 **Python MCP(Model Context Protocol) 서버**로 감싼 실습 코드.
Claude Desktop(무료)에서 자연어로 ① 모델을 학습하고 ② 적정 로봇 대수를 추정하고 ③ 모델 간 비교·시사점을 얻는다.

```
Claude Desktop ──stdio(MCP)──▶ server.py ──▶ fleet/train.py(학습) · fleet/registry.py(추론) ──▶ models/
```

**실습 순서 (약 40분, 모두 무료)** → 자세한 단계는 **[GUIDE.md](GUIDE.md)**

| 단계 | 할 일 | 확인 |
|---|---|---|
| 1. GitHub 연동 | VS Code ▸ `Git: Clone` ▸ 저장소 URL ▸ `rmfs-fleet-mcp` 폴더 열기 ▸ 터미널에서 `pip install -r requirements.txt` | 설치 완료 메시지 |
| 2. Claude 연동 | Claude Desktop 완전 종료(트레이 ▸ Quit) → `python tools/make_config.py --write` (설치 방식별 설정 파일 위치 자동 탐색) → 재실행 | 도구 아이콘에 `rmfs-fleet` |
| 3. 딥러닝 추정 | GUIDE.md 의 프롬프트 P1~P5 를 순서대로 입력 | 학습 성능표 · 추정 결과 · 모델 비교 · 시사점 |

## 데이터와 입력 형식

| 파일 | 내용 |
|---|---|
| `data/train_data.csv` | 학습 데이터 — 논문 공개 데이터셋 11,232건 (입력 열 + `TARGET` = Makespan 최소 로봇 대수) |
| `data/Data_single_row.csv` | 입력 형식 — 운영 조건 1건 = 12개 열. 도구 인자 이름도 이 열 이름과 같다 |

```
ORDER_NUM,SKU_COUNT_OPTIONS,NUM_AISLES,NUM_CROSS_AISLES,NUM_PS,PLACEMENT_CODE,BATCHING_MODE,OPTION,DEDICATED_MODE,MAX_ROBOTS,MAKESPAN_AT_MAX_ROBOTS,NUM_PODS
700,2,15,16,4,BL,SHARED_CLUSTER,1,pooled,40,2919,2176
```

| 열 | 의미 | 학습 범위 · 허용값 |
|---|---|---|
| `ORDER_NUM` | wave 당 주문 수 | 700 ~ 1,300 |
| `SKU_COUNT_OPTIONS` | 주문당 SKU 수 | 2 |
| `NUM_AISLES` · `NUM_CROSS_AISLES` | 통로 수 · 교차통로 수 | 15×15 (Square) · 12×18 (General) · 9×27 (Elongated) |
| `NUM_PS` | 피킹 스테이션 수 | 4 ~ 12 |
| `PLACEMENT_CODE` | 피킹 스테이션 배치 | BL · BS · OL · OS (B=Both sides, O=One side / L=Long side, S=Short side) |
| `BATCHING_MODE` | 주문 할당 | FCFS · SHARED_CLUSTER |
| `OPTION` | SKU-Pod-위치 전략 | 1 · 2 |
| `DEDICATED_MODE` | 로봇 할당 유연성 | dedicated · pooled |
| `MAX_ROBOTS` | 보유(가용) 로봇 대수 | NUM_PS × 10 |
| `MAKESPAN_AT_MAX_ROBOTS` | 보유 로봇 전량 투입 시 Makespan (초) | 876 ~ 9,228 |
| `NUM_PODS` | Pod 수 | 1,976 ~ 2,800 |

모델은 이 12개 열에서 계산한 13개 Feature(수치형 8 + 범주형 5, one-hot 후 21열 — 논문 Table 2)로 학습한다 (`fleet/features.py`).

## 도구 (Claude 가 호출)

| 도구 | 역할 |
|---|---|
| `get_dataset_info` | 학습 데이터(`data/train_data.csv`) 요약 |
| `train_models` | 논문 4.2절 설정으로 6개 모델 학습·평가·저장 (`alpha` · `hidden` · `dropout` · `lr` 로 튜닝) |
| `list_models` · `get_model_info` | 학습된 모델 목록·성능(R²·MAE·a20), 입력 열 정의·학습 범위 |
| `predict_fleet_size` | 운영 조건 12개 열 → 적정 로봇 대수 (`model` 로 선택) |
| `compare_models` | 같은 조건을 6개 모델로 추정해 한 표로 비교 |
| `whatif_order_volume` | 같은 조건에서 주문 수만 700~1,300 으로 바꿔 적정 대수 변화 추정 |
| `run_experiment_from_csv` · `describe_layout` | (심화) 여러 행 CSV 일괄 추정 · 레이아웃 개략도 SVG |

## 파일

```
server.py            MCP 서버 (도구 9 · 리소스 3 · 프롬프트 2)
tools/make_config.py Claude Desktop 설정 위치 탐색·저장        tools/train_models.py  터미널에서 전체 학습
fleet/features.py    입력 12개 열 · 검증 · 파생 Feature        fleet/train.py         학습 (논문 설정)
fleet/registry.py    모델 로딩·추론                            fleet/layout.py        레이아웃 SVG
data/train_data.csv  학습 데이터 (11,232건)                    data/Data_single_row.csv  입력 형식 예시 (1행)
models/              학습 결과물 (registry.json · *.joblib · *_state.pt · MODEL_CARD.md)
run_local.py         (참고) Claude 없이 학습+추정 확인         client_demo.py         (참고) Python MCP 클라이언트 예시
```

`models/` 에는 `data/train_data.csv` 를 논문 설정으로 학습한 결과가 들어 있어 clone 직후 바로 추정할 수 있다 (성능은 `models/MODEL_CARD.md`).

요구 사항: Python 3.10 이상 (3.13 에서 동작 확인) · Git · VS Code · Claude Desktop (무료 계정) — 모두 무료.
