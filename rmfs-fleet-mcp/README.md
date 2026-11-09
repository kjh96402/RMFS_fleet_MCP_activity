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
| 1. GitHub 연동 | VS Code ▸ `Git: Clone` ▸ 저장소 URL ▸ 폴더 열기 ▸ 터미널에서 `pip install -r requirements.txt` | 설치 완료 메시지 |
| 2. Python 작동 | `python run_local.py` (학습 + 추정 콘솔 출력) → `python client_demo.py` (Python MCP 클라이언트로 서버 연결) | `OK — MCP 연결 정상` |
| 3. Claude 연동 | `python tools/make_config.py --write` → Claude Desktop 완전 종료 후 재실행 | 도구 아이콘에 `rmfs-fleet` |
| 4. 딥러닝 추정 | GUIDE.md 의 프롬프트 P1~P5 를 순서대로 입력 | 학습 성능표 · 추정 결과 · 모델 비교 · 시사점 |

## 도구 (Claude 가 호출)

| 도구 | 역할 |
|---|---|
| `get_dataset_info` | 저장소 데이터셋(`data/dataset.csv`, 없으면 `dataset_sample.csv`) 요약 |
| `train_models` | 논문 4.2절 설정으로 6개 모델 학습·평가·저장 (`alpha` · `hidden` · `dropout` · `lr` 로 튜닝) |
| `list_models` · `get_model_info` | 학습된 모델 목록·성능(R²·MAE·a20), 입력 Feature 정의·허용 범위 |
| `predict_fleet_size` | 운영 조건 12개 → 적정 로봇 대수 (`model` 로 선택) |
| `compare_models` | 같은 조건을 6개 모델로 추정해 한 표로 비교 |
| `whatif_order_volume` | 주문 수 700~1,300 변화에 따른 적정 대수 |
| `run_experiment_from_csv` · `describe_layout` | (심화) CSV 여러 센터 일괄 실험 · 레이아웃 개략도 SVG |

## 파일

```
server.py          MCP 서버 (도구 정의)             run_local.py      Python 작동 확인 (학습+추정)
client_demo.py     Python MCP 클라이언트 연결 확인   tools/make_config.py  Claude Desktop 설정 생성
fleet/train.py     학습 (논문 설정·COLUMN_MAP)      fleet/features.py  입력 스키마·검증·파생 Feature
fleet/registry.py  모델 로딩·추론                   fleet/layout.py    레이아웃 SVG
data/dataset.csv   (발표자가 넣는 실제 데이터)       data/dataset_sample.csv  합성 샘플 (연습용)
models/            학습 결과물 (registry.json 등)    deploy/            (선택) 원격 배포용 Render 설정·HTTP 점검
```

## 발표자(저장소 소유자) 준비
1. `data/dataset.csv` 에 시뮬레이션 데이터셋을 넣는다 (열 이름이 다르면 `fleet/train.py` 의 `COLUMN_MAP` 수정).
2. `python tools/train_models.py` 로 학습해 `models/` 를 채운 뒤 **GitHub 에 함께 push** (학생은 clone 즉시 추정 가능).
3. GUIDE.md 의 Part A 리허설 → 학생에게 저장소 URL 과 GUIDE.md 공유.

요구 사항: Python 3.10~3.12 · Git · VS Code · Claude Desktop (무료 계정) — 모두 무료.
