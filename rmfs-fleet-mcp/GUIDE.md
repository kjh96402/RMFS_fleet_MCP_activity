# 실습 가이드 — MCP 기반 딥러닝 추정 모델 연동 (Python SDK · VS Code · Claude Desktop)

> 목표: GitHub 에 올린 로봇 대수 추정 모델을 Python MCP 서버로 불러와, Claude 에서 자연어로
> **학습(논문 설정) → 추정(Feature 입력) → 모델 비교 → 시사점** 을 얻는다. 코드는 전부 복사-붙여넣기, 비용 0원.

---

## Part A. 발표자 준비 (수업 전 1회, 약 30분)

| 단계 | 할 일 |
|---|---|
| A-1 데이터 | `data/dataset.csv` 에 시뮬레이션 데이터셋 저장. 열 이름이 표준(`order_volume`, `num_picking_stations`, `num_pods`, `max_robots`, `makespan_at_max_robots`, `skus_per_order`, `skus_per_pod`, 범주형 5개, `optimal_robots`)과 다르면 `fleet/train.py` 상단 `COLUMN_MAP`/`VALUE_MAP` 에 매핑 |
| A-2 학습 | 터미널: `python tools/train_models.py` → 콘솔의 R²·MAE·a20 을 논문 Table 5 와 비교. `models/` 에 결과물 생성 |
| A-3 점검 | `python run_local.py` → `python client_demo.py` 두 스크립트가 `OK` 로 끝나는지 확인 |
| A-4 GitHub | GitHub ▸ New repository(`rmfs-fleet-mcp`, Public) 생성 후 터미널: `git init` → `git add .` → `git commit -m "RMFS fleet-sizing MCP"` → `git branch -M main` → `git remote add origin https://github.com/<ID>/rmfs-fleet-mcp.git` → `git push -u origin main` (`models/` 포함, `dataset.csv` 는 `.gitignore` 로 제외됨 — 학생도 학습을 해 보게 하려면 `.gitignore` 의 `data/dataset*.csv` 줄을 지우고 push) |
| A-5 리허설 | Part B 를 처음부터 끝까지 한 번 수행하고 P1~P5 결과를 캡처 (슬라이드 기대 결과 표 채우기) |

학생에게 공유: 저장소 URL · 이 GUIDE.md · 프롬프트 P1~P5

---

## Part B. 학생 실습 (약 40분)

### Step 0. 준비물 (모두 무료)
- **Python 3.10~3.12** (python.org · 설치 시 *Add python.exe to PATH* 체크)
- **Git** (git-scm.com) · **VS Code** (code.visualstudio.com) + 확장 *Python*
- **Claude Desktop** (claude.ai/download) + 무료 계정 로그인

### Step 1. GitHub 연동 — VS Code 로 저장소 가져오기
1. VS Code 실행 → `Ctrl+Shift+P` → **Git: Clone** → 저장소 URL 붙여넣기 → 저장 폴더 선택 → **Open**
2. 메뉴 **Terminal ▸ New Terminal** (`` Ctrl+` ``) 에서:
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # 2~5분 (torch CPU 버전 포함)
```
3. 오른쪽 아래 인터프리터가 `.venv` 로 표시되는지 확인 (아니면 `Ctrl+Shift+P` ▸ **Python: Select Interpreter** ▸ `.venv`)

### Step 2. Python 작동 — 모델과 MCP 서버가 돌아가는지 확인
```bash
python run_local.py      # ① 데이터셋 확인 → ② 6개 모델 학습 → ③ 대표 시나리오 추정표  (약 1분)
python client_demo.py    # Python 이 MCP 클라이언트가 되어 server.py 에 연결 → 도구 목록 → 호출 결과
```
기대 출력(마지막 줄): `OK — Python 환경 정상` / `OK — MCP 연결 정상`

### Step 3. Claude Desktop 연동
```bash
python tools/make_config.py --write
```
→ 설정 파일(`%APPDATA%\Claude\claude_desktop_config.json`)에 아래 내용이 자동 저장된다 (직접 붙여넣어도 됨).
```json
{ "mcpServers": { "rmfs-fleet": { "command": "C:/…/rmfs-fleet-mcp/.venv/Scripts/python.exe",
                                   "args": ["C:/…/rmfs-fleet-mcp/server.py", "--stdio"] } } }
```
Claude Desktop 을 **완전히 종료**(작업 표시줄 트레이 아이콘 ▸ Quit) 후 다시 실행 → 입력창의 도구 아이콘에 **rmfs-fleet** 표시.
수동 편집 경로: Claude 메뉴 ▸ **Settings ▸ Developer ▸ Edit Config**

### Step 4. 딥러닝 추정 실습 — 프롬프트를 그대로 입력
**P1 · 데이터·도구 확인**
> rmfs-fleet 서버의 도구 목록을 보여 주고, get_dataset_info 로 데이터셋의 행 수·Feature·타깃 범위를 표로 정리해 줘. 아직 학습·예측은 하지 마.

**P2 · 모델 학습 (논문 설정 그대로)**
> train_models 를 기본값(논문 4.2절 설정: 수치형 z-score + 범주형 one-hot, 90/10 분할, MLR·SVR(RBF C=1)·RF(200트리 leaf 3)·XGB(200트리 depth 5)·MLP(256·BatchNorm·ReLU·Dropout 0.3·AdamW 1e-3)·C-MLP α=1.50)으로 실행하고, 모델별 R²·MAE·a20·학습 시간을 표로 정리한 뒤 논문 Table 5(MLP R² 0.954, MAE 3.92)와 비교해 줘.

**P3 · 파라미터 튜닝 (논문의 α 민감도 분석 방식)**
> train_models 를 models=["cmlp"] 로 alpha 를 1.25, 1.5, 2.0, 3.0 네 번 실행해 MAE·R² 변화를 표로 정리하고, 과소추정 페널티 α 를 키우면 무엇이 좋아지고 무엇이 나빠지는지 2줄로 설명해 줘. 마지막에는 alpha=1.5 로 다시 학습해 기본 상태로 돌려 줘.

**P4 · Feature 입력 → 적정 로봇 대수 추정**
> 피킹 스테이션 8개, Pod 2,176개, 보유 로봇 80대, 전체 투입 시 Makespan 4,400초, 이번 wave 주문 1,000건(주문당 SKU 2개, Pod당 SKU 20종), 통로 general, 스테이션 long_both, 주문 할당 shared_sku_clustering, SKU 배치 same_class_zoning, 로봇 pooled 조건에서 C-MLP 로 적정 로봇 대수를 추정하고, 절감 대수·warnings·mode 를 함께 보여 줘. 이어서 whatif_order_volume 로 주문 700~1,300건(100건 간격) 변화를 표와 꺾은선 차트로 보여 줘.

**P5 · 모델 비교 → 시사점**
> 같은 조건을 compare_models 로 MLR·SVR·RF·XGB·MLP·C-MLP 전부 추정해 추천 대수·절감 대수·테스트 성능을 한 표로 정리해 줘. 그리고 (1) 선형·커널·배깅·부스팅·딥러닝 결과가 왜 다른지, (2) 운영 보고에는 어떤 모델을 쓸지와 그 근거, (3) 추정치를 현장에 적용할 때의 유의점(학습 범위, 안전 여유, 사람의 최종 결정)을 각 2줄씩 시사점으로 정리해 줘.

(선택) **P6 · 학습 범위 밖 입력 검증** — 주문 2,000건으로 다시 추정해 warnings 를 그대로 보여 주고 보고서에 써도 되는지 판단해 줘.

### Step 5. 제출물 (최종 결과물)
1. P2 학습 성능표 캡처   2. P3 α 튜닝표 캡처   3. P4 추정 결과 + What-if 차트 캡처
4. P5 모델 비교표 + **시사점 3가지** (본인 문장으로 다시 정리)   5. 사용한 프롬프트 목록

---

## Part C. 막힐 때

| 증상 | 대응 |
|---|---|
| `pip install` 에서 torch 오류 | Python 3.10~3.12 인지 확인 (`python --version`), 64-bit, 재시도 |
| `python run_local.py` 가 `ModuleNotFoundError` | 가상환경 활성화 안 됨 → `.venv\Scripts\activate` 후 재실행 |
| Claude 도구 아이콘에 rmfs-fleet 없음 | Claude Desktop 을 트레이에서 **Quit** 후 재실행 · `make_config.py` 가 출력한 경로에 파일이 있는지 확인 |
| 도구 호출 시 `mode: demo` | 모델 미학습 → P2(train_models) 실행 또는 `python run_local.py` |
| `model='RF'` 오류 | 모델 키는 소문자 `rf` |
| 범주값 오류 (`'pool'`) | 오류 메시지의 허용값(`dedicated | pooled` …)으로 Claude 에게 재호출 요청 |
| 학습이 오래 걸림 | `quick=True`(기본)인지 확인 · 전체 설정(`quick=False`)은 수 분 소요 |
| (대안 호스트) VS Code Copilot Chat | `.vscode/mcp.json` 이 포함되어 있어 Agent 모드에서 같은 도구 사용 가능 |
