# 실습 가이드 — MCP 기반 딥러닝 추정 모델 연동 (Python SDK · VS Code · Claude Desktop)

> 목표: GitHub 에 올린 로봇 대수 추정 모델을 Python MCP 서버로 불러와, Claude 에서 자연어로
> **학습(논문 설정) → 추정(운영 조건 입력) → 모델 비교 → 시사점** 을 얻는다. 코드는 전부 복사-붙여넣기, 비용 0원.

저장소에 들어 있는 것: 학습 데이터 `data/train_data.csv`(논문 데이터 11,232건) · 입력 형식 `data/Data_single_row.csv`(12개 열) · 학습된 모델 `models/`

---

## 사전 설치 (모두 무료)
- **Python 3.10 이상** (python.org · 설치 시 *Add python.exe to PATH* 체크)
- **Git** (git-scm.com) · **VS Code** (code.visualstudio.com) + 확장 *Python*
- **Claude Desktop** (claude.ai/download) + 무료 계정 로그인

## Step 1. GitHub 연동 — VS Code 로 저장소 가져오기
1. VS Code 실행 → `Ctrl+Shift+P` → **Git: Clone** → 저장소 URL 붙여넣기 → 저장 폴더 선택
2. **File ▸ Open Folder** → `RMFS_fleet_MCP_activity` 안의 **`rmfs-fleet-mcp`** 폴더 열기
3. 메뉴 **Terminal ▸ New Terminal** (`` Ctrl+` ``) 에서 한 줄씩:
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # 2~5분 (torch CPU 버전 포함)
```
4. 오른쪽 아래 인터프리터가 `.venv` 로 표시되는지 확인 (아니면 `Ctrl+Shift+P` ▸ **Python: Select Interpreter** ▸ `.venv`)

## Step 2. Claude Desktop 연동
1. Claude Desktop 을 **완전히 종료** (작업 표시줄 트레이 아이콘 ▸ **Quit**) — 실행 중이면 앱이 설정 파일을 다시 덮어쓴다
2. VS Code 터미널(가상환경 활성화 상태)에서:
```bash
python tools/make_config.py          # 감지된 설정 파일 위치만 확인 (저장 안 함)
python tools/make_config.py --write  # 감지된 위치에 rmfs-fleet 항목 병합 저장 (기존 설정 유지 · .bak 백업)
```
3. Claude Desktop 다시 실행 → **Settings ▸ Developer** 에서 rmfs-fleet 이 `running` 인지 확인 → 입력창 도구 아이콘에 **rmfs-fleet** 표시

저장되는 내용 (경로는 본인 PC 기준으로 자동 입력):
```json
{ "mcpServers": { "rmfs-fleet": { "command": "C:/…/rmfs-fleet-mcp/.venv/Scripts/python.exe",
                                   "args": ["C:/…/rmfs-fleet-mcp/server.py", "--stdio"] } } }
```

**설정 파일 위치 — 설치 방식마다 다르다 (Windows)**

| 설치 방식 | Claude Desktop 이 읽는 파일 |
|---|---|
| Microsoft Store 빌드 (최신) | `%LOCALAPPDATA%\Claude-Data\claude_<패키지ID>\claude_desktop_config.json` |
| Microsoft Store 빌드 (이전) | `%LOCALAPPDATA%\Packages\Claude_<패키지ID>\LocalCache\Roaming\Claude\claude_desktop_config.json` |
| 일반 설치판 (.exe) | `%APPDATA%\Claude\claude_desktop_config.json` |

`make_config.py` 는 PC 에 있는 위치를 모두 찾아 각각 저장한다. Store 빌드는 `%APPDATA%\Claude` 를 읽지 않으므로 그 파일만 고치면 rmfs-fleet 가 나타나지 않는다.
직접 지정: `python tools/make_config.py --write --path "<설정 파일 전체 경로>"`

## Step 3. 로봇 대수 추정 실습 — 프롬프트를 그대로 입력
Claude Desktop 새 대화창(입력창 [+] 에서 rmfs-fleet 켜짐 확인)에 P1 → P5 를 순서대로 붙여넣는다.

**P1 · 데이터·도구 확인**
> rmfs-fleet 서버의 도구 목록을 보여 주고, get_dataset_info 로 데이터셋의 행 수·Feature·타깃 범위를 표로 정리해 줘. 아직 학습·예측은 하지 마.

**P2 · 모델 학습 (논문 설정 그대로)**
> train_models 를 기본값(논문 4.2절 설정: 수치형 z-score + 범주형 one-hot, 90/10 분할, MLR·SVR(RBF C=1)·RF(200트리 leaf 3)·XGB(200트리 depth 5)·MLP(256·BatchNorm·ReLU·Dropout 0.3·AdamW 1e-3)·C-MLP α=1.50)으로 실행하고, 모델별 R²·MAE·a20·학습 시간을 표로 정리한 뒤 논문 Table 5(MLP R² 0.954, MAE 3.92)와 비교해 줘.

**P3 · 파라미터 튜닝 (논문의 α 민감도 분석 방식)**
> train_models 를 models=["cmlp"] 로 alpha 를 1.25, 1.5, 2.0, 3.0 네 번 실행해 MAE·R² 변화를 표로 정리하고, 과소추정 페널티 α 를 키우면 무엇이 좋아지고 무엇이 나빠지는지 2줄로 설명해 줘. 마지막에는 alpha=1.5 로 다시 학습해 기본 상태로 돌려 줘.

**P4 · 운영 조건 입력 → 적정 로봇 대수 추정** (입력 = `data/Data_single_row.csv` 형식 1행)
> 아래 CSV 1행(Data_single_row.csv 형식)을 운영 조건으로 C-MLP 로 적정 로봇 대수를 추정하고, 절감 대수·warnings·mode 를 함께 보여 줘. 이어서 whatif_order_volume 로 주문 700~1,300건(100건 간격) 변화를 표와 꺾은선 차트로 보여 줘.
> ORDER_NUM,SKU_COUNT_OPTIONS,NUM_AISLES,NUM_CROSS_AISLES,NUM_PS,PLACEMENT_CODE,BATCHING_MODE,OPTION,DEDICATED_MODE,MAX_ROBOTS,MAKESPAN_AT_MAX_ROBOTS,NUM_PODS
> 700,2,15,16,4,BL,SHARED_CLUSTER,1,pooled,40,2919,2176

**P5 · 모델 비교 → 시사점**
> 같은 조건을 compare_models 로 MLR·SVR·RF·XGB·MLP·C-MLP 전부 추정해 추천 대수·절감 대수·테스트 성능을 한 표로 정리해 줘. 그리고 (1) 선형·커널·배깅·부스팅·딥러닝 결과가 왜 다른지, (2) 운영 보고에는 어떤 모델을 쓸지와 그 근거, (3) 추정치를 현장에 적용할 때의 유의점(학습 범위, 안전 여유, 사람의 최종 결정)을 각 2줄씩 시사점으로 정리해 줘.

(선택) **P6 · 학습 범위 밖 입력 검증** — 같은 조건에서 ORDER_NUM 만 2000 으로 바꿔 다시 추정해 warnings 를 그대로 보여 주고 보고서에 써도 되는지 판단해 줘.

**입력 12개 열** (P4 의 값을 자기 조건으로 바꿔 넣기 · 허용값은 `get_model_info` 로도 확인 가능)

| 열 | 의미 | 예시 | 학습 범위 · 허용값 |
|---|---|---|---|
| `ORDER_NUM` | wave 당 주문 수 | 700 | 700 ~ 1,300 |
| `SKU_COUNT_OPTIONS` | 주문당 SKU 수 | 2 | 2 |
| `NUM_AISLES` · `NUM_CROSS_AISLES` | 통로 수 · 교차통로 수 | 15 · 16 | 15×15 · 12×18 · 9×27 |
| `NUM_PS` | 피킹 스테이션 수 | 4 | 4 ~ 12 |
| `PLACEMENT_CODE` | 피킹 스테이션 배치 | BL | BL · BS · OL · OS |
| `BATCHING_MODE` | 주문 할당 | SHARED_CLUSTER | FCFS · SHARED_CLUSTER |
| `OPTION` | SKU-Pod-위치 전략 | 1 | 1 · 2 |
| `DEDICATED_MODE` | 로봇 할당 유연성 | pooled | dedicated · pooled |
| `MAX_ROBOTS` | 보유(가용) 로봇 대수 | 40 | NUM_PS × 10 |
| `MAKESPAN_AT_MAX_ROBOTS` | 전량 투입 시 Makespan (초) | 2919 | 876 ~ 9,228 |
| `NUM_PODS` | Pod 수 | 2176 | 1,976 ~ 2,800 |

## 제출물 (최종 결과물)
1. P2 학습 성능표 캡처   2. P3 α 튜닝표 캡처   3. P4 추정 결과 + What-if 차트 캡처
4. P5 모델 비교표 + **시사점 3가지** (본인 문장으로 다시 정리)   5. 사용한 프롬프트 목록

---

## 막힐 때

| 증상 | 대응 |
|---|---|
| `pip install` 에서 torch 오류 | 64-bit Python 인지 확인 (`python --version`), pip 업그레이드 후 재시도 |
| `ModuleNotFoundError` | 가상환경 활성화 안 됨 → `.venv\Scripts\activate` 후 재실행 |
| Claude 도구 아이콘에 rmfs-fleet 없음 | ① 트레이에서 **Quit** → `python tools/make_config.py` 로 감지 위치 확인 → `--write` 재실행 → Claude 재실행 ② Store 빌드인데 `%APPDATA%\Claude` 파일만 고친 경우: 앱이 읽는 파일은 `%LOCALAPPDATA%\Claude-Data\claude_…` 쪽 (Step 2 표) |
| `make_config.py` 가 "Claude Desktop 이 실행 중" 으로 중단 | 트레이 아이콘 ▸ Quit 으로 완전히 종료 후 다시 실행 (창만 닫으면 백그라운드에 남음) |
| "JSON 형식 오류 — 건너뜀" | 해당 파일을 메모장으로 열어 쉼표·괄호를 고치거나, `.bak` 백업으로 되돌린 뒤 재실행 |
| 도구 호출 시 `mode: demo` | 모델 파일 없음 → P2(train_models) 실행 |
| 범주값 오류 (`PLACEMENT_CODE='XX'` 등) | 오류 메시지의 허용값(`BL · BS · OL · OS` …)으로 Claude 에게 재호출 요청 |
| "필수 입력 누락" | 12개 열을 모두 입력 (`data/Data_single_row.csv` 의 헤더 그대로) |
| 학습이 오래 걸림 | `quick=True`(기본)인지 확인 · 전체 설정(`quick=False`, MLP 120 epoch)은 1~3분 소요 |

## (참고) Claude 없이 Python 으로 확인 — 실습에는 필요 없음
```bash
python run_local.py           # train_data.csv 확인 → 6개 모델 학습(quick) → Data_single_row.csv 조건 추정표
python client_demo.py         # Python 이 MCP 클라이언트가 되어 server.py 에 연결 → 도구 목록 → 호출 결과
python tools/train_models.py  # 논문 설정 전체 학습 (models/ 갱신)
```
`run_local.py` 는 quick 설정으로 다시 학습해 `models/` 를 덮어쓴다. 저장소의 전체 학습 결과로 되돌리려면 `python tools/train_models.py`.
