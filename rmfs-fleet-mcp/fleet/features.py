"""
features.py — 입력 형식(data/Data_single_row.csv), 검증(OOD), 파생 Feature(논문 Table 2 · 3.2.1)
=================================================================================================
서버·학습 스크립트가 공유하는 단일 정의.

입력  = data/Data_single_row.csv 의 12개 열 (학습 데이터 data/train_data.csv 와 같은 열 이름)
모델  = 수치형 8개 + 범주형 5개 = 13개 Feature (one-hot 후 21열, 논문 Table 2)
타깃  = TARGET (Makespan 최소 로봇 대수)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 입력 12개 열 (Data_single_row.csv 순서)
INPUT_COLUMNS = [
    "ORDER_NUM",                # wave 당 주문 수
    "SKU_COUNT_OPTIONS",        # 주문당 SKU 수 (학습 데이터는 2 고정)
    "NUM_AISLES",               # 통로 수
    "NUM_CROSS_AISLES",         # 교차통로 수
    "NUM_PS",                   # 피킹 스테이션 수
    "PLACEMENT_CODE",           # 피킹 스테이션 배치: BL · BS · OL · OS (B=Both sides, O=One side / L=Long side, S=Short side)
    "BATCHING_MODE",            # 주문 할당: FCFS · SHARED_CLUSTER
    "OPTION",                   # SKU-Pod-위치 전략: 1 · 2 (논문 Table 1 의 2개 수준, 학습 데이터 코드값)
    "DEDICATED_MODE",           # 로봇 할당 유연성: dedicated · pooled
    "MAX_ROBOTS",               # 보유(가용) 로봇 대수 (학습 데이터는 NUM_PS × 10)
    "MAKESPAN_AT_MAX_ROBOTS",   # 보유 로봇 전량 투입 시 Makespan (초)
    "NUM_PODS",                 # Pod 수
]
INTEGER_COLUMNS = ["ORDER_NUM", "SKU_COUNT_OPTIONS", "NUM_AISLES", "NUM_CROSS_AISLES", "NUM_PS", "OPTION", "MAX_ROBOTS", "NUM_PODS"]
TARGET = "TARGET"

# 입력 범주형 4개의 허용값 (학습 데이터 표기 그대로)
INPUT_CATEGORICAL = {
    "PLACEMENT_CODE": ["BL", "BS", "OL", "OS"],
    "BATCHING_MODE": ["FCFS", "SHARED_CLUSTER"],
    "OPTION": [1, 2],
    "DEDICATED_MODE": ["dedicated", "pooled"],
}
# 통로 구성(논문 Table 1): (NUM_AISLES, NUM_CROSS_AISLES) → AISLE_CONFIGURATION (종횡비 = 교차통로 / 통로)
AISLE_LAYOUTS = {"Square": (15, 15), "General": (12, 18), "Elongated": (9, 27)}

# ------------------------------------------------------------------ 모델 Feature 13개 (논문 Table 2)
NUMERIC_FEATURES = [
    "ORDER_NUM", "NUM_PS", "WAREHOUSE_DIM", "MAX_ROBOTS", "MAKESPAN_AT_MAX_ROBOTS",
    "ROBOT_DENSITY", "REQUIRED_THROUGHPUT_PER_PS_HOUR", "REQUIRED_FULFILL_PER_POD_HOUR",
]
CATEGORICAL = {   # 허용값 순서 = one-hot 열 순서
    "AISLE_CONFIGURATION": list(AISLE_LAYOUTS),
    **INPUT_CATEGORICAL,
}
CATEGORICAL_FEATURES = list(CATEGORICAL)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 학습 데이터(train_data.csv, 11,232건)의 입력 범위 — 범위 밖 입력은 warnings 로 표시
TRAIN_RANGES = {
    "ORDER_NUM": (700, 1300),
    "NUM_PS": (4, 12),
    "NUM_PODS": (1976, 2800),
    "MAKESPAN_AT_MAX_ROBOTS": (876, 9228),
}


def aisle_configuration(num_aisles, num_cross_aisles):
    """(통로 수, 교차통로 수) → Square · General · Elongated. 학습 조합이 아니면 종횡비가 가장 가까운 구성."""
    ratio = np.asarray(num_cross_aisles, dtype=float) / np.asarray(num_aisles, dtype=float)
    names = list(AISLE_LAYOUTS)
    ref = np.array([c / a for a, c in AISLE_LAYOUTS.values()])
    idx = np.abs(np.log(ratio[..., None]) - np.log(ref)).argmin(axis=-1)
    return np.array(names, dtype=object)[idx]


def engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """입력 12개 열 → 모델 Feature 13개 (파생 4개 계산). 학습(train_data.csv)과 추정이 같은 식을 쓴다."""
    out = pd.DataFrame(index=df.index)
    out["ORDER_NUM"] = df["ORDER_NUM"].astype(float)
    out["NUM_PS"] = df["NUM_PS"].astype(float)
    out["WAREHOUSE_DIM"] = df["NUM_PODS"] / df["NUM_PS"]
    out["MAX_ROBOTS"] = df["MAX_ROBOTS"].astype(float)
    out["MAKESPAN_AT_MAX_ROBOTS"] = df["MAKESPAN_AT_MAX_ROBOTS"].astype(float)
    out["ROBOT_DENSITY"] = df["MAX_ROBOTS"] / df["NUM_PODS"]
    out["REQUIRED_THROUGHPUT_PER_PS_HOUR"] = (df["ORDER_NUM"] * df["SKU_COUNT_OPTIONS"]) / (df["NUM_PS"] * df["MAKESPAN_AT_MAX_ROBOTS"])
    out["REQUIRED_FULFILL_PER_POD_HOUR"] = df["ORDER_NUM"] / (df["NUM_PODS"] * df["MAKESPAN_AT_MAX_ROBOTS"])
    out["AISLE_CONFIGURATION"] = pd.Series(aisle_configuration(df["NUM_AISLES"].to_numpy(), df["NUM_CROSS_AISLES"].to_numpy()),
                                           index=df.index, dtype=object)
    out["PLACEMENT_CODE"] = df["PLACEMENT_CODE"].astype(object)
    out["BATCHING_MODE"] = df["BATCHING_MODE"].astype(object)
    out["OPTION"] = df["OPTION"].astype(int)
    out["DEDICATED_MODE"] = df["DEDICATED_MODE"].astype(object)
    return out[ALL_FEATURES]


def engineer(s: dict) -> dict:
    """입력 1건(dict) → 모델 Feature 13개(dict)."""
    row = engineer_frame(pd.DataFrame([s])[INPUT_COLUMNS]).iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def to_frame(feature_rows: list[dict]) -> pd.DataFrame:
    """Feature dict 목록 → 모델 입력 DataFrame (열 순서 고정)."""
    return pd.DataFrame(feature_rows)[ALL_FEATURES]


def normalize_scenario(raw: dict) -> dict:
    """CSV 행·도구 인자 → 열 이름·표기 정리 및 숫자형 변환 (대소문자·공백 차이 허용)."""
    s = {str(k).strip().upper(): v for k, v in raw.items() if k is not None and v not in (None, "")}
    missing = [c for c in INPUT_COLUMNS if c not in s]
    if missing:
        raise ValueError(f"필수 입력 누락: {missing} (입력 형식: data/Data_single_row.csv 의 12개 열)")
    for c in INTEGER_COLUMNS:
        s[c] = int(float(s[c]))
    s["MAKESPAN_AT_MAX_ROBOTS"] = float(s["MAKESPAN_AT_MAX_ROBOTS"])
    s["PLACEMENT_CODE"] = str(s["PLACEMENT_CODE"]).strip().upper()
    s["BATCHING_MODE"] = str(s["BATCHING_MODE"]).strip().upper()
    s["DEDICATED_MODE"] = str(s["DEDICATED_MODE"]).strip().lower()
    return {c: s[c] for c in INPUT_COLUMNS}


def validate(s: dict) -> list[str]:
    """범주값·양수 검사(오류) 및 학습 범위 밖 입력 경고(OOD) 반환. s 는 normalize_scenario 결과."""
    for c, cats in INPUT_CATEGORICAL.items():
        if s[c] not in cats:
            raise ValueError(f"{c}={s[c]!r} 는 허용값 {cats} 중 하나여야 함")
    for c in ("ORDER_NUM", "SKU_COUNT_OPTIONS", "NUM_AISLES", "NUM_CROSS_AISLES", "NUM_PS", "MAX_ROBOTS", "MAKESPAN_AT_MAX_ROBOTS", "NUM_PODS"):
        if s[c] <= 0:
            raise ValueError(f"{c}={s[c]} 는 0 보다 커야 함")
    w = []
    for c, (lo, hi) in TRAIN_RANGES.items():
        if not (lo <= s[c] <= hi):
            w.append(f"{c}={s[c]} 는 학습 범위 [{lo}, {hi}] 밖 (OOD, 신뢰도 저하 가능)")
    if s["MAX_ROBOTS"] != 10 * s["NUM_PS"]:
        w.append(f"MAX_ROBOTS={s['MAX_ROBOTS']} 는 학습 조건(NUM_PS × 10 = {10 * s['NUM_PS']})과 다름 (OOD)")
    if s["SKU_COUNT_OPTIONS"] != 2:
        w.append(f"SKU_COUNT_OPTIONS={s['SKU_COUNT_OPTIONS']} 는 학습 조건(2)과 다름 (OOD)")
    cfg = str(aisle_configuration(s["NUM_AISLES"], s["NUM_CROSS_AISLES"]))
    a, c_ = AISLE_LAYOUTS[cfg]
    ratio, ref = s["NUM_CROSS_AISLES"] / s["NUM_AISLES"], c_ / a
    if abs(ratio / ref - 1) > 0.15 or abs(s["NUM_AISLES"] - a) > 3:
        w.append(f"통로 구성 {s['NUM_AISLES']}×{s['NUM_CROSS_AISLES']} 는 학습 구성(15×15 · 12×18 · 9×27)과 달라 "
                 f"가장 가까운 {cfg} 로 처리함 (OOD)")
    return w
