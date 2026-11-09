"""
RMFS 로봇 대수 추정 MCP 서버 (다중 모델 · CSV 실험 · 레이아웃)
=================================================================
Kim, Kang & Jung (2026), Computers & Industrial Engineering 214:111910 의 시뮬레이션-ML 프레임워크에서
학습한 비교 모델군(MLR · SVR · RF · XGB · MLP · C-MLP)을 MCP(Model Context Protocol) 도구로 노출한다.

Tools      get_dataset_info · train_models · list_models · get_model_info · predict_fleet_size · compare_models
           whatif_order_volume · run_experiment_from_csv · describe_layout
Resources  scenario://template · model://card · models://registry
Prompts    fleet_sizing_report · model_comparison_report

실행   python server.py            → Streamable HTTP  http://<host>:8000/mcp  (Claude 사용자 지정 커넥터)
       python server.py --stdio    → stdio            (Claude Desktop 로컬 연결)
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys

import numpy as np
from mcp.server.mcpserver import MCPServer

from fleet.features import CATEGORICAL, RAW_NUMERIC, engineer, normalize_scenario, to_frame, validate
from fleet.layout import layout_spec, layout_svg
from fleet.registry import ModelRegistry
from fleet.train import ALL_MODELS, PAPER_HP, dataset_info, train_all

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "models"))
REG = ModelRegistry(MODEL_DIR)

mcp = MCPServer(
    name="rmfs-fleet-sizing",
    instructions=(
        "RMFS(Robotic Mobile Fulfillment System) 운영 시나리오를 입력받아 Makespan 최소화 기준 적정 로봇 대수를 "
        "추정하는 도구 모음. 순서: get_dataset_info → train_models(논문 설정) → list_models → predict_fleet_size / compare_models. "
        "모델은 model 인자로 선택(기본 cmlp). "
        "학습 범위 밖 입력은 warnings 로 반환되므로 결과 보고 시 반드시 mode 와 warnings 를 함께 제시할 것. "
        "레이아웃 그림이 필요하면 describe_layout 의 svg 를 그대로 렌더링할 것."
    ),
)


# ------------------------------------------------------------------ 내부 공통
def _predict_one(s: dict, model: str | None) -> dict:
    s = normalize_scenario(s)
    warnings = validate(s)
    feats = engineer(s)
    key = REG.resolve(model)
    y = float(REG.predict(key, to_frame([feats]))[0])
    rec = int(max(1, min(round(y), s["max_robots"])))                 # 정수 반올림 · [1, max_robots] clip
    if REG.mode == "demo":
        warnings.append("mode=demo: 학습 모델이 없어 휴리스틱으로 응답함 (연결 테스트 전용)")
    return {
        "mode": REG.mode, "model": key, "recommended_robots": rec, "max_robots": s["max_robots"],
        "robot_savings": s["max_robots"] - rec, "raw_prediction": round(y, 2),
        "engineered_features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.items()},
        "warnings": warnings,
    }


def _scenario(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


# ------------------------------------------------------------------ Tools
@mcp.tool()
def get_dataset_info() -> dict:
    """저장소에 포함된 학습 데이터셋(data/dataset.csv, 없으면 dataset_sample.csv)의 행 수·Feature·타깃 범위·범주값을 반환. 학습 전에 먼저 호출."""
    return dataset_info()


@mcp.tool()
def train_models(
    models: list[str] | None = None, alpha: float = 1.5, hidden: int = 256, dropout: float = 0.3, lr: float = 0.001,
    quick: bool = True,
) -> dict:
    """저장소 데이터셋으로 로봇 대수 추정 모델군을 학습·평가·저장하고 모델별 테스트 성능(R²·MAE·a20)을 반환.

    기본값은 논문(Kim, Kang & Jung, 2026) 4.2절 설정과 동일: 수치형 z-score + 범주형 one-hot(21열), 90/10 분할,
    MLR · SVR(RBF, C=1) · RF(200트리, leaf 3) · XGB(200트리, depth 5) · MLP(256, BN·ReLU·Dropout 0.3, AdamW 1e-3) · C-MLP(α=1.50).

    Args:
        models: 학습할 모델 키 목록 (mlr · svr · rf · xgb · mlp · cmlp). 비우면 6개 전부
        alpha: C-MLP 비대칭 손실의 과소추정 페널티 계수 (논문 기본 1.50, 민감도 분석 범위 1.25~3.0)
        hidden: MLP 은닉 뉴런 수 (논문 최종 256) · dropout: Dropout 비율 (0.3) · lr: 학습률 (1e-3)
        quick: True(기본)면 MLP 60 epoch·트리 50개로 빠르게(수십 초), False 면 논문 설정 전체(수 분)
    Returns: dataset, hyperparameters, models{key: metrics·train_seconds}, 학습 후 서버 모델이 즉시 교체됨
    """
    global REG
    reg = train_all(None, models or ALL_MODELS, {"alpha": alpha, "hidden": hidden, "dropout": dropout, "lr": lr},
                    quick=quick, out_dir=MODEL_DIR, log=lambda *_: None)
    REG = ModelRegistry(MODEL_DIR)
    return {"dataset": reg["dataset"], "n_train": reg["n_train"], "n_test": reg["n_test"], "quick": quick,
            "hyperparameters": {k: reg["hyperparameters"][k] for k in ("alpha", "hidden", "dropout", "lr", "batch_size")},
            "models": {k: {"metrics": v["metrics"], "train_seconds": v["train_seconds"]} for k, v in reg["models"].items()
                       if k in (models or ALL_MODELS)},
            "default": reg["default"], "mode": REG.mode,
            "note": "metrics 는 테스트셋(10 %) 기준, MAE 는 정수 반올림 후 로봇 대수 오차. 논문 Table 5: MLP R² 0.954 · MAE 3.92"}


@mcp.tool()
def list_models() -> dict:
    """사용 가능한 추정 모델 목록(MLR·SVR·RF·XGB·MLP·C-MLP)과 테스트 성능(R²·MAE·a20), 기본 모델, 현재 mode(trained/demo)."""
    return {"mode": REG.mode, "default": REG.default, "models": REG.describe(),
            "load_errors": REG.meta.get("load_errors", {}), "trained_at": REG.meta.get("trained_at")}


@mcp.tool()
def get_model_info(model: str = "default") -> dict:
    """선택 모델의 설명과 입력 Feature 정의·허용 범위(논문 Table 1), 출처, 유의사항을 반환."""
    key = REG.resolve(model)
    m = REG.models[key]
    return {
        "mode": REG.mode, "model": key, "name": m.name, "type": m.kind, "metrics_test": m.metrics,
        "source": "Kim, Kang & Jung (2026). A simulation-based deep learning approach to operational robot fleet sizing "
                  "in robotic mobile fulfillment systems. Computers & Industrial Engineering, 214, 111910. "
                  "https://doi.org/10.1016/j.cie.2026.111910",
        "target": "Makespan 최소화 기준 적정 로봇 대수 (정수)",
        "training_data": "DES(SimPy) 11,232 시나리오 × 10 반복 × 41 fleet 수준 = 4,605,120 runs (논문 4.1절)",
        "inputs": {"numeric_ranges": RAW_NUMERIC, "categorical": CATEGORICAL},
        "note": "입력이 학습 범위 밖이면 warnings 에 표시. 최종 투입 대수는 관리자가 안전 여유(safety buffer)를 더해 결정.",
    }


@mcp.tool()
def predict_fleet_size(
    order_volume: int, num_picking_stations: int, num_pods: int, max_robots: int, makespan_at_max_robots: float,
    skus_per_order: float = 2.0, skus_per_pod: float = 20.0, aisle_configuration: str = "general",
    picking_station_placement: str = "long_both", order_assignment: str = "shared_sku_clustering",
    sku_pod_location: str = "same_class_zoning", robot_assignment_flexibility: str = "pooled",
    model: str = "default",
) -> dict:
    """단일 운영 시나리오(1 wave)의 적정 로봇 대수를 선택 모델로 추정.

    Args:
        order_volume: wave 당 주문 수 (학습 범위 700~1300)
        num_picking_stations: 피킹 스테이션 수 (4~12)
        num_pods: Pod 수
        max_robots: 보유(가용) 로봇 최대 대수 (학습 범위: 스테이션 수 × 5 ~ × 10)
        makespan_at_max_robots: 전체 로봇 투입 시 Makespan (초)
        skus_per_order: 주문당 평균 SKU 수 (1~3) · skus_per_pod: Pod당 SKU 종류 수 (15~25)
        aisle_configuration: square | general | elongated
        picking_station_placement: long_both | long_one | short_both | short_one
        order_assignment: fcfs | shared_sku_clustering · sku_pod_location: class_random_scatter | same_class_zoning
        robot_assignment_flexibility: dedicated | pooled
        model: mlr | svr | rf | xgb | mlp | cmlp | default(=cmlp)
    """
    args = dict(locals())
    return _predict_one(_scenario(**{k: v for k, v in args.items() if k != "model"}), model)


@mcp.tool()
def compare_models(
    order_volume: int, num_picking_stations: int, num_pods: int, max_robots: int, makespan_at_max_robots: float,
    skus_per_order: float = 2.0, skus_per_pod: float = 20.0, aisle_configuration: str = "general",
    picking_station_placement: str = "long_both", order_assignment: str = "shared_sku_clustering",
    sku_pod_location: str = "same_class_zoning", robot_assignment_flexibility: str = "pooled",
) -> dict:
    """같은 시나리오를 모든 모델(MLR·SVR·RF·XGB·MLP·C-MLP)로 추정해 모델별 추천 대수·절감·테스트 성능을 한 표로 반환."""
    s = normalize_scenario(_scenario(**dict(locals())))
    rows = []
    for key in REG.models:
        r = _predict_one(s, key)
        rows.append({"model": key, "name": REG.models[key].name, "recommended_robots": r["recommended_robots"],
                     "robot_savings": r["robot_savings"], "raw_prediction": r["raw_prediction"],
                     "metrics_test": REG.models[key].metrics})
    return {"mode": REG.mode, "max_robots": s["max_robots"], "warnings": validate(s), "rows": rows,
            "note": "모델 간 차이는 학습 알고리즘 차이에서 비롯됨. 논문 Table 5 기준 MLP(C-MLP)가 R²·MAE·a20 최고."}


@mcp.tool()
def whatif_order_volume(
    num_picking_stations: int, num_pods: int, max_robots: int, makespan_at_max_robots: float,
    start: int = 700, stop: int = 1300, step: int = 50, model: str = "default",
    skus_per_order: float = 2.0, skus_per_pod: float = 20.0, aisle_configuration: str = "general",
    picking_station_placement: str = "long_both", order_assignment: str = "shared_sku_clustering",
    sku_pod_location: str = "same_class_zoning", robot_assignment_flexibility: str = "pooled",
) -> dict:
    """주문 수를 start~stop 범위에서 step 간격으로 바꿔 가며 적정 로봇 대수 변화를 일괄 추정(What-if). 다른 조건은 고정."""
    args = dict(locals())
    fixed = {k: v for k, v in args.items() if k not in ("start", "stop", "step", "model")}
    rows = []
    for ov in range(start, stop + 1, step):
        r = _predict_one({**fixed, "order_volume": ov}, model)
        rows.append({"order_volume": ov, "recommended_robots": r["recommended_robots"], "robot_savings": r["robot_savings"],
                     "warnings": r["warnings"]})
    return {"mode": REG.mode, "model": REG.resolve(model), "fixed": {"num_picking_stations": num_picking_stations,
            "max_robots": max_robots}, "rows": rows}


@mcp.tool()
def run_experiment_from_csv(csv_text: str, model: str = "default", include_layout: bool = False) -> dict:
    """scenario://template 형식 CSV(여러 행 = 여러 물류센터/wave)를 받아 행별 적정 로봇 대수·절감 효과를 일괄 산출(실험 결과표).

    Args:
        csv_text: 헤더 포함 CSV 텍스트. 선택 열 center_id(센터 이름). 누락 열은 기본값 적용.
        model: 사용할 모델 키 (list_models 참고) 또는 'all'(모든 모델로 각 행 추정)
        include_layout: True 면 행별 레이아웃 스펙(통로·블록·스테이션 위치) 포함
    Returns: results(행별), summary(평균 절감 대수·절감률), warnings_count
    """
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    keys = list(REG.models) if model == "all" else [REG.resolve(model)]
    results, savings = [], []
    for i, row in enumerate(reader, start=1):
        cid = row.pop("center_id", None) or f"row{i}"
        row = {k.strip(): v for k, v in row.items() if k}
        entry = {"row": i, "center_id": cid}
        try:
            s = normalize_scenario(row)
            preds = {k: _predict_one(s, k) for k in keys}
            first = preds[keys[0]]
            entry.update({"order_volume": s["order_volume"], "num_picking_stations": s["num_picking_stations"],
                          "max_robots": s["max_robots"], "warnings": first["warnings"],
                          "predictions": {k: {"recommended_robots": p["recommended_robots"], "robot_savings": p["robot_savings"]}
                                          for k, p in preds.items()}})
            savings.append(first["robot_savings"] / s["max_robots"])
            if include_layout:
                entry["layout"] = layout_spec(s["aisle_configuration"], s["num_picking_stations"],
                                              s["picking_station_placement"], s["num_pods"], s["max_robots"])
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
def describe_layout(
    aisle_configuration: str = "general", num_picking_stations: int = 8, picking_station_placement: str = "long_both",
    num_pods: int | None = None, max_robots: int | None = None, title: str = "RMFS 레이아웃 개략도",
) -> dict:
    """물류센터 레이아웃 개략도: 통로 구성(논문 Table 1)에 따른 블록 격자·Pod 용량·점유율·스테이션 위치(spec)와 SVG 그림(svg).

    svg 문자열은 그대로 HTML/아티팩트로 렌더링 가능. 종횡비 = 교차통로 수 / 통로 수 (square 1.0 · general 1.5 · elongated 3.0).
    """
    if aisle_configuration not in CATEGORICAL["aisle_configuration"]:
        raise ValueError(f"aisle_configuration 허용값: {CATEGORICAL['aisle_configuration']}")
    if picking_station_placement not in CATEGORICAL["picking_station_placement"]:
        raise ValueError(f"picking_station_placement 허용값: {CATEGORICAL['picking_station_placement']}")
    spec = layout_spec(aisle_configuration, num_picking_stations, picking_station_placement, num_pods, max_robots)
    return {"spec": spec, "svg": layout_svg(spec, title)}


# ------------------------------------------------------------------ Resources
@mcp.resource("scenario://template", mime_type="text/csv")
def scenario_template() -> str:
    """입력 CSV 템플릿 (헤더 + 예시 3행: 센터 A·B·C). run_experiment_from_csv 의 입력 형식."""
    return open(os.path.join(BASE, "data", "template.csv"), encoding="utf-8").read()


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
def fleet_sizing_report(num_picking_stations: int = 8, max_robots: int = 80, model: str = "cmlp") -> str:
    """관리자용 로봇 운영 보고서 작성 프롬프트 (모델 확인 → 예측 → What-if → 검증 → 보고)."""
    return (
        f"피킹 스테이션 {num_picking_stations}개, 보유 로봇 {max_robots}대인 RMFS 물류센터의 이번 wave 적정 로봇 대수 보고서를 작성해 줘.\n"
        f"1) list_models 로 사용 가능한 모델과 mode 를 확인하고 {model} 모델을 사용해 줘.\n"
        "2) predict_fleet_size 로 기본 시나리오(주문 1,000건)를 추정해 줘.\n"
        "3) whatif_order_volume 로 주문 700~1,300건 변화에 따른 적정 대수를 표로 정리해 줘.\n"
        "4) describe_layout 으로 레이아웃 개략도를 받아 그림으로 보여 줘.\n"
        "5) warnings 와 mode 를 반드시 표기하고, 추정치는 권고일 뿐 최종 투입 대수는 관리자가 안전 여유를 더해 결정한다고 명시해 줘."
    )


@mcp.prompt()
def model_comparison_report(order_volume: int = 1000, num_picking_stations: int = 8, max_robots: int = 80) -> str:
    """모델 비교 보고서 프롬프트: 같은 시나리오를 MLR·SVR·RF·XGB·MLP·C-MLP 로 추정해 차이와 성능을 해석."""
    return (
        f"주문 {order_volume}건, 피킹 스테이션 {num_picking_stations}개, 보유 로봇 {max_robots}대 시나리오를 compare_models 로 "
        "모든 모델에 대해 추정해 줘. 모델별 추천 대수·절감 대수·테스트 성능(R²·MAE·a20)을 한 표로 정리하고, "
        "선형(MLR)·커널(SVR)·배깅(RF)·부스팅(XGB)·딥러닝(MLP/C-MLP)의 차이가 왜 생기는지 3줄로 설명해 줘. "
        "운영 보고에는 어떤 모델을 쓸지 근거와 함께 추천해 줘."
    )


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), stateless_http=True)
