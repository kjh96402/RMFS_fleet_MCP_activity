"""
RMFS 로봇 대수 추정 MCP 서버 (다중 모델 · CSV 실험 · 레이아웃)
=================================================================
Kim, Kang & Jung (2026), Computers & Industrial Engineering 214:111910 의 시뮬레이션-ML 프레임워크에서
학습한 비교 모델군(MLR · SVR · RF · XGB · MLP · C-MLP)을 MCP(Model Context Protocol) 도구로 노출한다.

Tools      get_dataset_info · train_models · list_models · get_model_info · predict_fleet_size · compare_models
           whatif_order_volume · run_experiment_from_csv · describe_layout
Resources  scenario://template · model://card · models://registry
Prompts    fleet_sizing_report · model_comparison_report

데이터     data/train_data.csv        학습 데이터 (논문 공개 데이터셋 11,232건)
입력 형식  data/Data_single_row.csv   운영 조건 1건 = 12개 열 (도구 인자 이름 = 열 이름)

실행   python server.py --stdio    → stdio (Claude Desktop 이 자동 실행 · 본 실습 방식)
       python server.py            → Streamable HTTP  http://<host>:8000/mcp  (원격 연결용, 실습에서는 사용 안 함)
"""
from __future__ import annotations

import csv
import functools
import io
import json
import os
import sys

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from fleet.features import AISLE_LAYOUTS, INPUT_CATEGORICAL, INPUT_COLUMNS, TRAIN_RANGES, engineer, normalize_scenario, to_frame, validate
from fleet.layout import layout_spec, layout_svg
from fleet.registry import ModelRegistry
from fleet.train import ALL_MODELS, dataset_info, train_all

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "models"))
TEMPLATE_CSV = os.path.join(BASE, "data", "Data_single_row.csv")
REG = ModelRegistry(MODEL_DIR)

mcp = MCPServer(
    name="rmfs-fleet-sizing",
    instructions=(
        "RMFS(Robotic Mobile Fulfillment System) 운영 조건을 입력받아 Makespan 최소화 기준 적정 로봇 대수를 "
        "추정하는 도구 모음. 순서: get_dataset_info → train_models(논문 설정) → list_models → predict_fleet_size / compare_models. "
        "입력은 data/Data_single_row.csv 의 12개 열(ORDER_NUM, SKU_COUNT_OPTIONS, NUM_AISLES, NUM_CROSS_AISLES, NUM_PS, "
        "PLACEMENT_CODE, BATCHING_MODE, OPTION, DEDICATED_MODE, MAX_ROBOTS, MAKESPAN_AT_MAX_ROBOTS, NUM_PODS)이며 "
        "도구 인자 이름도 같다. 모델은 model 인자로 선택(기본 cmlp). "
        "학습 범위 밖 입력은 warnings 로 반환되므로 결과 보고 시 반드시 mode 와 warnings 를 함께 제시할 것. "
        "레이아웃 그림이 필요하면 describe_layout 의 svg 를 그대로 렌더링할 것."
    ),
)


# ------------------------------------------------------------------ 내부 공통
def _input_errors(fn):
    """입력 오류(허용되지 않는 범주값·누락 열 등)를 ToolError 로 바꿔, 오류 메시지가 Claude 에 전달되어 스스로 고쳐 재호출하게 한다."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, FileNotFoundError) as e:
            raise ToolError(str(e)) from e
    return wrapper


def _predict_one(raw: dict, model: str | None) -> dict:
    s = normalize_scenario(raw)
    warnings = validate(s)
    feats = engineer(s)
    key = REG.resolve(model)
    y = float(REG.predict(key, to_frame([feats]))[0])
    rec = int(max(1, min(round(y), s["MAX_ROBOTS"])))                 # 정수 반올림 · [1, MAX_ROBOTS] clip
    if REG.mode == "demo":
        warnings.append("mode=demo: 학습 모델이 없어 휴리스틱으로 응답함 (연결 테스트 전용)")
    return {
        "mode": REG.mode, "model": key, "recommended_robots": rec, "max_robots": s["MAX_ROBOTS"],
        "robot_savings": s["MAX_ROBOTS"] - rec, "raw_prediction": round(y, 2),
        "engineered_features": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in feats.items()},
        "warnings": warnings,
    }


# ------------------------------------------------------------------ Tools
@mcp.tool()
@_input_errors
def get_dataset_info() -> dict:
    """저장소의 학습 데이터셋(data/train_data.csv, 논문 데이터 11,232건)의 행 수·입력 열·모델 Feature·타깃 범위·범주값을 반환. 학습 전에 먼저 호출."""
    return dataset_info()


@mcp.tool()
@_input_errors
def train_models(
    models: list[str] | None = None, alpha: float = 1.5, hidden: int = 256, dropout: float = 0.3, lr: float = 0.001,
    quick: bool = True,
) -> dict:
    """data/train_data.csv 로 로봇 대수 추정 모델군을 학습·평가·저장하고 모델별 테스트 성능(R²·MAE·a20)을 반환.

    기본값은 논문(Kim, Kang & Jung, 2026) 4.2절 설정과 동일: 수치형 z-score + 범주형 one-hot(21열), 90/10 분할(10,108 / 1,124건),
    MLR · SVR(RBF, C=1) · RF(200트리, leaf 3) · XGB(200트리, depth 5) · MLP(256, BN·ReLU·Dropout 0.3, AdamW 1e-3, batch 64) · C-MLP(α=1.50).

    Args:
        models: 학습할 모델 키 목록 (mlr · svr · rf · xgb · mlp · cmlp). 비우면 6개 전부
        alpha: C-MLP 비대칭 손실의 과소추정 페널티 계수 (논문 기본 1.50, 민감도 분석 범위 1.25~3.0)
        hidden: MLP 은닉 뉴런 수 (논문 최종 256) · dropout: Dropout 비율 (0.3) · lr: 학습률 (1e-3)
        quick: True(기본)면 MLP 30 epoch·트리 50개로 빠르게(수십 초), False 면 논문 설정 전체(MLP 120 epoch·트리 200개, 수 분)
    Returns: dataset, hyperparameters, models{key: metrics·train_seconds}, 학습 후 서버 모델이 즉시 교체됨
    """
    global REG
    reg = train_all(None, models or ALL_MODELS, {"alpha": alpha, "hidden": hidden, "dropout": dropout, "lr": lr},
                    quick=quick, out_dir=MODEL_DIR, log=lambda *_: None)
    REG = ModelRegistry(MODEL_DIR)
    return {"dataset": reg["dataset"], "n_rows": reg["n_rows"], "n_train": reg["n_train"], "n_test": reg["n_test"], "quick": quick,
            "hyperparameters": {k: reg["hyperparameters"][k] for k in ("alpha", "hidden", "dropout", "lr", "batch_size", "epochs")},
            "models": {k: {"metrics": v["metrics"], "train_seconds": v["train_seconds"]} for k, v in reg["models"].items()
                       if k in (models or ALL_MODELS)},
            "default": reg["default"], "mode": REG.mode,
            "note": "metrics 는 테스트셋(10 %) 기준, MAE 는 정수 반올림 후 로봇 대수 오차. "
                    "논문 Table 5: MLR 0.951/4.09 · SVR 0.949/4.03 · RF 0.948/4.26 · XGB 0.947/4.16 · MLP 0.954/3.92 (R²/MAE)"}


@mcp.tool()
@_input_errors
def list_models() -> dict:
    """사용 가능한 추정 모델 목록(MLR·SVR·RF·XGB·MLP·C-MLP)과 테스트 성능(R²·MAE·a20), 기본 모델, 현재 mode(trained/demo)."""
    return {"mode": REG.mode, "default": REG.default, "models": REG.describe(), "dataset": REG.meta.get("dataset"),
            "quick": REG.meta.get("quick"), "load_errors": REG.meta.get("load_errors", {}), "trained_at": REG.meta.get("trained_at")}


@mcp.tool()
@_input_errors
def get_model_info(model: str = "default") -> dict:
    """선택 모델의 설명과 입력 12개 열(Data_single_row.csv 형식)·학습 범위·허용 범주값, 출처, 유의사항을 반환."""
    key = REG.resolve(model)
    m = REG.models[key]
    return {
        "mode": REG.mode, "model": key, "name": m.name, "type": m.kind, "metrics_test": m.metrics,
        "source": "Kim, Kang & Jung (2026). A simulation-based deep learning approach to operational robot fleet sizing "
                  "in robotic mobile fulfillment systems. Computers & Industrial Engineering, 214, 111910. "
                  "https://doi.org/10.1016/j.cie.2026.111910",
        "target": "TARGET — Makespan 최소화 기준 적정 로봇 대수 (정수)",
        "training_data": "data/train_data.csv — DES(SimPy) 11,232 시나리오 × 10 반복 × 41 fleet 수준 = 4,605,120 runs (논문 4.1절)",
        "inputs": {
            "columns": INPUT_COLUMNS,
            "train_ranges": {**{k: list(v) for k, v in TRAIN_RANGES.items()}, "MAX_ROBOTS": "NUM_PS × 10", "SKU_COUNT_OPTIONS": 2},
            "aisle_layouts(NUM_AISLES × NUM_CROSS_AISLES)": {k: list(v) for k, v in AISLE_LAYOUTS.items()},
            "categorical": INPUT_CATEGORICAL,
            "codes": {"PLACEMENT_CODE": "B=Both sides · O=One side / L=Long side · S=Short side",
                      "BATCHING_MODE": "FCFS=선착순 · SHARED_CLUSTER=Shared SKU clustering",
                      "OPTION": "SKU-Pod-위치 전략 2개 수준 (학습 데이터 코드값 1 · 2)",
                      "DEDICATED_MODE": "dedicated=스테이션 전담 · pooled=공용"},
        },
        "note": "입력이 학습 범위 밖이면 warnings 에 표시. 최종 투입 대수는 관리자가 안전 여유(safety buffer)를 더해 결정.",
    }


@mcp.tool()
@_input_errors
def predict_fleet_size(
    ORDER_NUM: int, SKU_COUNT_OPTIONS: int, NUM_AISLES: int, NUM_CROSS_AISLES: int, NUM_PS: int,
    PLACEMENT_CODE: str, BATCHING_MODE: str, OPTION: int, DEDICATED_MODE: str,
    MAX_ROBOTS: int, MAKESPAN_AT_MAX_ROBOTS: float, NUM_PODS: int,
    model: str = "default",
) -> dict:
    """운영 조건 1건(1 wave)의 적정 로봇 대수를 선택 모델로 추정. 인자 = data/Data_single_row.csv 의 12개 열.

    Args:
        ORDER_NUM: wave 당 주문 수 (학습 범위 700~1300)
        SKU_COUNT_OPTIONS: 주문당 SKU 수 (학습 데이터 2)
        NUM_AISLES: 통로 수 · NUM_CROSS_AISLES: 교차통로 수 (학습 구성 15×15 Square · 12×18 General · 9×27 Elongated)
        NUM_PS: 피킹 스테이션 수 (4~12)
        PLACEMENT_CODE: 스테이션 배치 BL | BS | OL | OS (B=Both sides, O=One side / L=Long side, S=Short side)
        BATCHING_MODE: 주문 할당 FCFS | SHARED_CLUSTER
        OPTION: SKU-Pod-위치 전략 1 | 2
        DEDICATED_MODE: 로봇 할당 dedicated | pooled
        MAX_ROBOTS: 보유(가용) 로봇 대수 (학습 조건: NUM_PS × 10)
        MAKESPAN_AT_MAX_ROBOTS: 보유 로봇 전량 투입 시 Makespan (초)
        NUM_PODS: Pod 수 (1976~2800)
        model: mlr | svr | rf | xgb | mlp | cmlp | default(=cmlp)
    Returns: recommended_robots(추천 대수) · robot_savings(절감 대수) · engineered_features · warnings · mode
    """
    args = dict(locals())
    return _predict_one({c: args[c] for c in INPUT_COLUMNS}, model)


@mcp.tool()
@_input_errors
def compare_models(
    ORDER_NUM: int, SKU_COUNT_OPTIONS: int, NUM_AISLES: int, NUM_CROSS_AISLES: int, NUM_PS: int,
    PLACEMENT_CODE: str, BATCHING_MODE: str, OPTION: int, DEDICATED_MODE: str,
    MAX_ROBOTS: int, MAKESPAN_AT_MAX_ROBOTS: float, NUM_PODS: int,
) -> dict:
    """같은 운영 조건(Data_single_row.csv 의 12개 열)을 모든 모델(MLR·SVR·RF·XGB·MLP·C-MLP)로 추정해 모델별 추천 대수·절감·테스트 성능을 한 표로 반환."""
    args = dict(locals())
    s = normalize_scenario({c: args[c] for c in INPUT_COLUMNS})
    rows = []
    for key in REG.models:
        r = _predict_one(s, key)
        rows.append({"model": key, "name": REG.models[key].name, "recommended_robots": r["recommended_robots"],
                     "robot_savings": r["robot_savings"], "raw_prediction": r["raw_prediction"],
                     "metrics_test": REG.models[key].metrics})
    return {"mode": REG.mode, "max_robots": s["MAX_ROBOTS"], "warnings": validate(s), "rows": rows,
            "note": "모델 간 차이는 학습 알고리즘 차이에서 비롯됨. 논문 Table 5 기준 MLP 가 R²·MAE·a20 최고, C-MLP(α=1.50)는 과소추정을 줄인 최종 모델."}


@mcp.tool()
@_input_errors
def whatif_order_volume(
    ORDER_NUM: int, SKU_COUNT_OPTIONS: int, NUM_AISLES: int, NUM_CROSS_AISLES: int, NUM_PS: int,
    PLACEMENT_CODE: str, BATCHING_MODE: str, OPTION: int, DEDICATED_MODE: str,
    MAX_ROBOTS: int, MAKESPAN_AT_MAX_ROBOTS: float, NUM_PODS: int,
    start: int = 700, stop: int = 1300, step: int = 50, scale_makespan: bool = True, model: str = "default",
) -> dict:
    """기준 운영 조건(Data_single_row.csv 의 12개 열)에서 주문 수(ORDER_NUM)만 start~stop 범위로 바꿔 가며 적정 로봇 대수 변화를 일괄 추정(What-if).

    Args:
        (12개 열): 기준 조건. predict_fleet_size 와 같은 인자
        start · stop · step: 바꿔 볼 주문 수 범위와 간격 (학습 범위 700~1300)
        scale_makespan: True(기본)면 MAKESPAN_AT_MAX_ROBOTS 를 주문 수에 비례해 환산(기준 Makespan × 주문 수 / 기준 ORDER_NUM).
            학습 데이터에서 같은 조건의 Makespan ÷ 주문 수 는 거의 일정(변동계수 약 2 %). False 면 Makespan 을 기준값으로 고정
        model: 사용할 모델 키 (기본 cmlp)
    """
    args = dict(locals())
    base = normalize_scenario({c: args[c] for c in INPUT_COLUMNS})
    rows = []
    for ov in range(start, stop + 1, step):
        mk = round(base["MAKESPAN_AT_MAX_ROBOTS"] * ov / base["ORDER_NUM"], 1) if scale_makespan else base["MAKESPAN_AT_MAX_ROBOTS"]
        r = _predict_one({**base, "ORDER_NUM": ov, "MAKESPAN_AT_MAX_ROBOTS": mk}, model)
        rows.append({"ORDER_NUM": ov, "MAKESPAN_AT_MAX_ROBOTS": mk, "recommended_robots": r["recommended_robots"],
                     "robot_savings": r["robot_savings"], "warnings": r["warnings"]})
    return {"mode": REG.mode, "model": REG.resolve(model), "base": base, "scale_makespan": scale_makespan, "rows": rows}


@mcp.tool()
@_input_errors
def run_experiment_from_csv(csv_text: str, model: str = "default", include_layout: bool = False) -> dict:
    """Data_single_row.csv 형식 CSV(헤더 + 여러 행 = 여러 물류센터/wave)를 받아 행별 적정 로봇 대수·절감 효과를 일괄 산출(실험 결과표).

    Args:
        csv_text: 헤더 포함 CSV 텍스트. 12개 열 필수(scenario://template 참고), 선택 열 center_id(센터 이름)
        model: 사용할 모델 키 (list_models 참고) 또는 'all'(모든 모델로 각 행 추정)
        include_layout: True 면 행별 레이아웃 스펙(통로·블록·스테이션 위치) 포함
    Returns: results(행별), summary(평균 절감 대수·절감률), warnings_count
    """
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    keys = list(REG.models) if model == "all" else [REG.resolve(model)]
    results, savings = [], []
    for i, row in enumerate(reader, start=1):
        row = {str(k).strip(): v for k, v in row.items() if k}
        cid = next((row.pop(k) for k in list(row) if k.lower() == "center_id"), None) or f"row{i}"
        entry = {"row": i, "center_id": cid}
        try:
            s = normalize_scenario(row)
            preds = {k: _predict_one(s, k) for k in keys}
            first = preds[keys[0]]
            entry.update({"ORDER_NUM": s["ORDER_NUM"], "NUM_PS": s["NUM_PS"], "MAX_ROBOTS": s["MAX_ROBOTS"],
                          "warnings": first["warnings"],
                          "predictions": {k: {"recommended_robots": p["recommended_robots"], "robot_savings": p["robot_savings"]}
                                          for k, p in preds.items()}})
            savings.append(first["robot_savings"] / s["MAX_ROBOTS"])
            if include_layout:
                entry["layout"] = layout_spec(s["NUM_AISLES"], s["NUM_CROSS_AISLES"], s["NUM_PS"], s["PLACEMENT_CODE"],
                                              s["NUM_PODS"], s["MAX_ROBOTS"])
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)
    ok = [r for r in results if "error" not in r]
    summary = {"n_rows": len(results), "n_ok": len(ok), "models": keys,
               "avg_robot_savings": round(float(np.mean([r["predictions"][keys[0]]["robot_savings"] for r in ok])), 2) if ok else None,
               "avg_savings_rate_pct": round(float(np.mean(savings)) * 100, 1) if savings else None}
    return {"mode": REG.mode, "summary": summary, "results": results,
            "warnings_count": sum(1 for r in ok if r["warnings"])}


@mcp.tool()
@_input_errors
def describe_layout(
    NUM_AISLES: int = 15, NUM_CROSS_AISLES: int = 15, NUM_PS: int = 4, PLACEMENT_CODE: str = "BL",
    NUM_PODS: int | None = None, MAX_ROBOTS: int | None = None, title: str = "RMFS 레이아웃 개략도",
) -> dict:
    """물류센터 레이아웃 개략도: 통로 구성(논문 Table 1)에 따른 블록 격자·Pod 용량·점유율·스테이션 위치(spec)와 SVG 그림(svg).

    svg 문자열은 그대로 HTML/아티팩트로 렌더링 가능. 종횡비 = NUM_CROSS_AISLES / NUM_AISLES (Square 1.0 · General 1.5 · Elongated 3.0).
    """
    code = str(PLACEMENT_CODE).strip().upper()
    if code not in INPUT_CATEGORICAL["PLACEMENT_CODE"]:
        raise ValueError(f"PLACEMENT_CODE 허용값: {INPUT_CATEGORICAL['PLACEMENT_CODE']}")
    if not (1 <= NUM_AISLES <= 40 and 1 <= NUM_CROSS_AISLES <= 60):
        raise ValueError("NUM_AISLES 는 1~40, NUM_CROSS_AISLES 는 1~60 범위여야 함")
    spec = layout_spec(NUM_AISLES, NUM_CROSS_AISLES, NUM_PS, code, NUM_PODS, MAX_ROBOTS)
    return {"spec": spec, "svg": layout_svg(spec, title)}


# ------------------------------------------------------------------ Resources
@mcp.resource("scenario://template", mime_type="text/csv")
def scenario_template() -> str:
    """입력 양식 = data/Data_single_row.csv (헤더 12개 열 + 예시 1행). predict_fleet_size 의 인자이자 run_experiment_from_csv 의 입력 형식."""
    return open(TEMPLATE_CSV, encoding="utf-8-sig").read()


@mcp.resource("model://card", mime_type="text/markdown")
def model_card() -> str:
    """모델 카드: 출처 논문, 학습 데이터 생성 방식, 모델군·성능, 사용 시 유의사항."""
    return open(os.path.join(BASE, "models", "MODEL_CARD.md"), encoding="utf-8").read()


@mcp.resource("models://registry", mime_type="application/json")
def models_registry() -> str:
    """현재 로딩된 모델 레지스트리(파일·성능·학습 시각) JSON."""
    return json.dumps({"mode": REG.mode, "default": REG.default, "models": REG.describe(), "meta": {
        k: v for k, v in REG.meta.items() if k != "models"}}, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ Prompts
@mcp.prompt()
def fleet_sizing_report(model: str = "cmlp") -> str:
    """관리자용 로봇 운영 보고서 작성 프롬프트 (모델 확인 → 예측 → What-if → 레이아웃 → 보고)."""
    return (
        "scenario://template (data/Data_single_row.csv) 의 운영 조건으로 이번 wave 적정 로봇 대수 보고서를 작성해 줘.\n"
        f"1) list_models 로 사용 가능한 모델과 mode 를 확인하고 {model} 모델을 사용해 줘.\n"
        "2) predict_fleet_size 에 12개 열 값을 그대로 넣어 적정 로봇 대수를 추정해 줘.\n"
        "3) whatif_order_volume 에 같은 12개 열을 넣어 ORDER_NUM 700~1,300 변화에 따른 적정 대수를 표로 정리해 줘.\n"
        "4) describe_layout 으로 레이아웃 개략도를 받아 그림으로 보여 줘.\n"
        "5) warnings 와 mode 를 반드시 표기하고, 추정치는 권고일 뿐 최종 투입 대수는 관리자가 안전 여유를 더해 결정한다고 명시해 줘."
    )


@mcp.prompt()
def model_comparison_report() -> str:
    """모델 비교 보고서 프롬프트: 같은 운영 조건을 MLR·SVR·RF·XGB·MLP·C-MLP 로 추정해 차이와 성능을 해석."""
    return (
        "scenario://template (data/Data_single_row.csv) 의 운영 조건을 compare_models 로 모든 모델에 대해 추정해 줘. "
        "모델별 추천 대수·절감 대수·테스트 성능(R²·MAE·a20)을 한 표로 정리하고, "
        "선형(MLR)·커널(SVR)·배깅(RF)·부스팅(XGB)·딥러닝(MLP/C-MLP)의 차이가 왜 생기는지 3줄로 설명해 줘. "
        "운영 보고에는 어떤 모델을 쓸지 근거와 함께 추천해 줘."
    )


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), stateless_http=True)
