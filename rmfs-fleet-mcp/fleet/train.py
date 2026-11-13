"""
train.py — data/train_data.csv → 비교 모델군 학습 · 평가 · 저장 (논문 4.2절 설정 그대로)
========================================================================================
  MLR   LinearRegression
  SVR   RBF, C=1.0, gamma='scale'
  RF    n_estimators=200, max_depth=None, min_samples_leaf=3
  XGB   n_estimators=200, max_depth=5
  MLP   Linear(21,256)-BatchNorm-ReLU-Dropout(0.3)-Linear(256,1), AdamW(lr 1e-3, wd 1e-4), batch 64,
        ReduceLROnPlateau(patience 10, factor 0.5), 학습셋 전체로 120 epoch
  C-MLP 위 구조 + 비대칭 손실 (과소추정 페널티 α=1.50, Eq. 1)
데이터: data/train_data.csv — 논문 공개 데이터셋 11,232건 (입력 12개 열 + TARGET, 열 이름은 fleet/features.py)
전처리: 수치형 8개 z-score + 범주형 5개 one-hot (21열) — sklearn ColumnTransformer (joblib 저장)
평가:   학습 90 % (10,108건) / 테스트 10 % (1,124건, random_state=42) · R² · MAE(정수 반올림 후) · a20-index

server.py 의 train_models 도구와 tools/train_models.py CLI 가 모두 이 모듈을 사용한다.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from fleet.features import (ALL_FEATURES, CATEGORICAL, CATEGORICAL_FEATURES, INPUT_CATEGORICAL, INPUT_COLUMNS,
                            NUMERIC_FEATURES, TARGET, engineer_frame)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_MODELS = ["mlr", "svr", "rf", "xgb", "mlp", "cmlp"]

# 논문 Table 4 · Section 4.2 기본 하이퍼파라미터 (train_models 도구의 기본값)
PAPER_HP = {"hidden": 256, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 120, "alpha": 1.5,
            "rf_n_estimators": 200, "rf_min_samples_leaf": 3, "xgb_n_estimators": 200, "xgb_max_depth": 5, "svr_C": 1.0}

DATASET_NAME = "train_data.csv"
DATASET_PATH = os.path.join(ROOT, "data", DATASET_NAME)


def default_dataset_path() -> tuple[str, str]:
    """학습 데이터 = data/train_data.csv (논문 공개 데이터셋, 11,232 시나리오)."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"학습 데이터 없음: {DATASET_PATH} (저장소의 data/{DATASET_NAME} 가 있어야 함)")
    return DATASET_PATH, f"{DATASET_NAME} (논문 시뮬레이션 데이터셋)"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in [*INPUT_COLUMNS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"데이터셋 열 부족: {missing} (필요: 입력 12개 열 {INPUT_COLUMNS} + {TARGET})")
    for col, cats in INPUT_CATEGORICAL.items():
        bad = set(df[col].unique()) - set(cats)
        if bad:
            raise ValueError(f"{col} 에 허용되지 않는 값 {bad} (허용: {cats})")
    return df


def load_dataset(path: str):
    """CSV → (모델 Feature 13개 DataFrame, 타깃 배열). 파생 Feature 는 추정 때와 같은 engineer_frame 으로 계산."""
    df = load_raw(path)
    return engineer_frame(df[INPUT_COLUMNS]), df[TARGET].astype(float).to_numpy()


def dataset_info(path: str | None = None) -> dict:
    path, source = (path, os.path.basename(path)) if path else default_dataset_path()
    df = load_raw(path)
    X, y = engineer_frame(df[INPUT_COLUMNS]), df[TARGET].astype(float).to_numpy()
    num_in = [c for c in INPUT_COLUMNS if c not in INPUT_CATEGORICAL]
    return {"source": source, "path": path, "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
            "input_columns": INPUT_COLUMNS, "model_features": ALL_FEATURES, "target": TARGET,
            "target_stats": {"min": int(y.min()), "max": int(y.max()), "mean": round(float(y.mean()), 1)},
            "input_ranges": {c: [round(float(df[c].min()), 2), round(float(df[c].max()), 2)] for c in num_in},
            "input_categorical_values": {c: sorted(df[c].unique().tolist()) for c in INPUT_CATEGORICAL},
            "engineered_ranges": {c: [round(float(X[c].min()), 6), round(float(X[c].max()), 6)]
                                  for c in ("WAREHOUSE_DIM", "ROBOT_DENSITY", "REQUIRED_THROUGHPUT_PER_PS_HOUR", "REQUIRED_FULFILL_PER_POD_HOUR")},
            "aisle_configuration_counts": X["AISLE_CONFIGURATION"].value_counts().to_dict(),
            "note": "입력 형식은 data/Data_single_row.csv (12개 열). 모델은 여기서 파생한 13개 Feature(one-hot 후 21열)로 학습."}


def make_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[CATEGORICAL[c] for c in CATEGORICAL_FEATURES], handle_unknown="error"), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def metrics(y, yhat) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    yr = np.rint(yhat)
    return {"R2": round(float(r2_score(y, yr)), 4), "MAE": round(float(mean_absolute_error(y, yr)), 3),
            "a20_index": round(float(np.mean(np.abs(yr - y) <= 0.2 * y)), 3)}


def train_sklearn(key, Xtr, ytr, hp, quick):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR

    if key == "mlr":
        reg = LinearRegression()
    elif key == "svr":
        reg = SVR(kernel="rbf", C=hp["svr_C"], gamma="scale")
    elif key == "rf":
        reg = RandomForestRegressor(n_estimators=50 if quick else hp["rf_n_estimators"], max_depth=None,
                                    min_samples_leaf=hp["rf_min_samples_leaf"], random_state=42, n_jobs=-1)
    elif key == "xgb":
        from xgboost import XGBRegressor

        reg = XGBRegressor(n_estimators=50 if quick else hp["xgb_n_estimators"], max_depth=hp["xgb_max_depth"], random_state=42, n_jobs=-1)
    else:
        raise ValueError(key)
    pipe = Pipeline([("pre", make_preprocessor()), ("reg", reg)])
    pipe.fit(Xtr, ytr)
    return pipe


def train_torch(Xtr_np, ytr, hp, alpha, quick, seed=42):
    """MLP(α=1) · C-MLP(α>1) 학습. 논문 4.2절: 선택된 구조를 학습셋 전체로 120 epoch 재학습 (quick 은 30 epoch)."""
    import torch

    from fleet.registry import build_mlp

    torch.manual_seed(seed)
    n = len(Xtr_np)
    X = torch.tensor(Xtr_np, dtype=torch.float32); y = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    net = build_mlp(X.shape[1], hp["hidden"], hp["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    def loss_fn(pred, true):                                    # Eq. (1): 과소추정(pred < true)에 α 배 페널티
        se = (true - pred) ** 2
        return (torch.where(pred < true, torch.full_like(se, alpha), torch.ones_like(se)) * se).mean()

    epochs, bs = (30 if quick else int(hp["epochs"])), int(hp["batch_size"])   # quick: Claude 도구 호출 안에 끝나는 연습용
    for ep in range(epochs):
        net.train()
        perm = np.random.RandomState(seed + ep).permutation(n)
        total = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(); loss = loss_fn(net(X[b]), y[b]); loss.backward(); opt.step()
            total += loss.item() * len(b)
        sched.step(total / n)
    net.eval()
    return net, epochs


def train_all(dataset_path: str | None = None, models: list[str] | None = None, hp: dict | None = None,
              quick: bool = False, out_dir: str | None = None, log=print) -> dict:
    """모델군 학습 → models/registry.json 저장 → 레지스트리 dict 반환."""
    from sklearn.model_selection import train_test_split

    hp = {**PAPER_HP, **(hp or {})}
    models = [m for m in (models or ALL_MODELS)]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise ValueError(f"알 수 없는 모델 키 {bad} (선택: {ALL_MODELS})")
    path, source = (dataset_path, os.path.basename(dataset_path)) if dataset_path else default_dataset_path()
    out_dir = out_dir or os.path.join(ROOT, "models")
    X, y = load_dataset(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.10, random_state=42)
    os.makedirs(out_dir, exist_ok=True)
    reg = {"source": "Kim, Kang & Jung (2026) C&IE 214:111910 — Section 4.2 settings", "dataset": source, "n_rows": int(len(X)),
           "default": "cmlp" if "cmlp" in models else models[0], "hyperparameters": hp, "quick": quick,
           "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "trained_at": time.strftime("%Y-%m-%d %H:%M"),
           "features": ALL_FEATURES, "models": {}}
    log(f"데이터 {source}: {len(X):,}건 → 학습 {len(Xtr):,} / 테스트 {len(Xte):,}")
    pre = None
    for key in models:
        t0 = time.time()
        if key in ("mlr", "svr", "rf", "xgb"):
            pipe = train_sklearn(key, Xtr, ytr, hp, quick)
            m = metrics(yte, pipe.predict(Xte))
            joblib.dump(pipe, os.path.join(out_dir, f"{key}.joblib"), compress=3)
            reg["models"][key] = {"type": "sklearn", "file": f"{key}.joblib", "metrics": m}
        else:
            if pre is None:
                pre = make_preprocessor().fit(Xtr)
                joblib.dump(pre, os.path.join(out_dir, "preprocess.joblib"))
            alpha = hp["alpha"] if key == "cmlp" else 1.0
            net, epochs = train_torch(pre.transform(Xtr), ytr, hp, alpha, quick)
            import torch

            with torch.no_grad():
                yhat = net(torch.tensor(pre.transform(Xte), dtype=torch.float32)).squeeze(1).numpy()
            m = metrics(yte, yhat); m["epochs"] = epochs
            torch.save(net.state_dict(), os.path.join(out_dir, f"{key}_state.pt"))
            reg["models"][key] = {"type": "torch", "preprocess": "preprocess.joblib", "state": f"{key}_state.pt",
                                  "n_features": int(pre.transform(Xtr[:1]).shape[1]), "hidden": hp["hidden"],
                                  "dropout": hp["dropout"], "alpha": alpha, "metrics": m}
        reg["models"][key]["train_seconds"] = round(time.time() - t0, 1)
        log(f"  {key:5s}  R2={m['R2']:.3f}  MAE={m['MAE']:.2f}  a20={m['a20_index']:.2f}  ({reg['models'][key]['train_seconds']}s)")
    # 이전 레지스트리에 있던 다른 모델은 유지(부분 재학습 지원)
    prev_path = os.path.join(out_dir, "registry.json")
    if os.path.exists(prev_path):
        try:
            prev = json.load(open(prev_path, encoding="utf-8"))
            for k, v in prev.get("models", {}).items():
                if k not in reg["models"] and os.path.exists(os.path.join(out_dir, v.get("file") or v.get("state", ""))):
                    reg["models"][k] = v
        except Exception:
            pass
    json.dump(reg, open(prev_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"저장 완료 → {prev_path} (기본 모델: {reg['default']})")
    return reg
