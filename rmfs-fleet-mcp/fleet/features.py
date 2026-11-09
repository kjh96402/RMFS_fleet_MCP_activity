"""
features.py — 입력 스키마(논문 Table 1), 검증(OOD), 파생 Feature(논문 Table 2 · 3.2.1)
=====================================================================================
서버·학습 스크립트가 공유하는 단일 정의. 모델 입력 = 수치형 8개 + 범주형 5개 (one-hot 후 21열).
"""
from __future__ import annotations

import pandas as pd

# ------------------------------------------------------------------ 논문 Table 1 범위
RAW_NUMERIC = {               # 사용자가 입력하는 원본 수치
    "order_volume": (700, 1300),
    "num_picking_stations": (4, 12),
    "num_pods": (100, 5000),
    "max_robots": None,       # 스테이션 수 × 5 ~ × 10 (동적 검사)
    "makespan_at_max_robots": (1, None),
    "skus_per_order": (1, 3),
    "skus_per_pod": (15, 25),
}
CATEGORICAL = {               # 범주형 5개 (허용값 순서 = one-hot 열 순서)
    "aisle_configuration": ["square", "general", "elongated"],
    "picking_station_placement": ["long_both", "long_one", "short_both", "short_one"],
    "order_assignment": ["fcfs", "shared_sku_clustering"],
    "sku_pod_location": ["class_random_scatter", "same_class_zoning"],
    "robot_assignment_flexibility": ["dedicated", "pooled"],
}
NUMERIC_FEATURES = [          # 모델 수치형 입력 8개 (논문 Table 2 순서)
    "order_volume", "num_picking_stations", "warehouse_dimension", "current_num_robots",
    "current_makespan", "robot_density", "avg_throughput_per_station", "avg_fulfillment_per_pod",
]
CATEGORICAL_FEATURES = list(CATEGORICAL)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "optimal_robots"

DEFAULTS = dict(skus_per_order=2.0, skus_per_pod=20.0, aisle_configuration="general",
                picking_station_placement="long_both", order_assignment="shared_sku_clustering",
                sku_pod_location="same_class_zoning", robot_assignment_flexibility="pooled")


def engineer(s: dict) -> dict:
    """원본 입력 → 모델 Feature 13개 (파생 4개 계산, 논문 3.2.1)."""
    wd = (s["num_pods"] * s["skus_per_pod"]) / s["num_picking_stations"]        # warehouse dimension
    workload = s["order_volume"] * s["skus_per_order"]
    return {
        "order_volume": s["order_volume"],
        "num_picking_stations": s["num_picking_stations"],
        "warehouse_dimension": wd,
        "current_num_robots": s["max_robots"],
        "current_makespan": s["makespan_at_max_robots"],
        "robot_density": s["max_robots"] / wd,
        "avg_throughput_per_station": workload / (s["num_picking_stations"] * s["makespan_at_max_robots"]),
        "avg_fulfillment_per_pod": workload / (s["num_pods"] * s["makespan_at_max_robots"]),
        **{k: s[k] for k in CATEGORICAL_FEATURES},
    }


def validate(s: dict) -> list[str]:
    """필수값·범주값 검사(오류) 및 학습 범위 밖 입력 경고(OOD) 반환."""
    for k in RAW_NUMERIC:
        if k not in s or s[k] is None:
            raise ValueError(f"필수 입력 누락: {k}")
    for k, cats in CATEGORICAL.items():
        if s.get(k) not in cats:
            raise ValueError(f"{k}={s.get(k)!r} 는 허용값 {cats} 중 하나여야 함")
    w = []
    for k, rng in RAW_NUMERIC.items():
        if rng:
            lo, hi = rng
            if (lo is not None and s[k] < lo) or (hi is not None and s[k] > hi):
                w.append(f"{k}={s[k]} 는 학습 범위 {rng} 밖 (OOD, 신뢰도 저하 가능)")
    ps = s["num_picking_stations"]
    if not (5 * ps <= s["max_robots"] <= 10 * ps):
        w.append(f"max_robots={s['max_robots']} 는 학습 범위 [{5*ps}, {10*ps}] (스테이션 수×5~×10) 밖")
    return w


def to_frame(feature_rows: list[dict]) -> pd.DataFrame:
    """Feature dict 목록 → 모델 입력 DataFrame (열 순서 고정)."""
    return pd.DataFrame(feature_rows)[ALL_FEATURES]


def normalize_scenario(raw: dict) -> dict:
    """CSV 행·도구 인자 → 기본값 보완 및 숫자형 변환."""
    s = {**DEFAULTS, **{k: v for k, v in raw.items() if v not in (None, "")}}
    for k in RAW_NUMERIC:
        if k in s and not isinstance(s[k], (int, float)):
            f = float(s[k]); s[k] = int(f) if f.is_integer() else f
    for k in ("skus_per_order", "skus_per_pod"):
        s[k] = float(s[k])
    return s
