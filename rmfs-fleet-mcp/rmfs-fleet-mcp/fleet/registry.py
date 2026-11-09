"""
registry.py — 학습된 모델 레지스트리 (models/registry.json)
============================================================
tools/train_models.py 가 생성한 모델들을 읽어 공통 인터페이스 predict(model_key, X_df) 로 제공한다.

  sklearn 계열 (MLR · SVR · RF · XGB) : Pipeline(전처리 + 회귀기)  → *.joblib
  torch 계열  (MLP · C-MLP)          : preprocess.joblib + *_state.pt (논문 Table 4 구조 재생성)
레지스트리가 없으면 demo 모드(휴리스틱 1개)로 동작하며 모든 응답에 mode 가 표시된다.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

MODEL_FAMILY = {   # 논문 4.2절 비교 모델군
    "mlr": "선형 회귀 (Multiple Linear Regression)",
    "svr": "커널 회귀 (Support Vector Regression, RBF)",
    "rf": "배깅 앙상블 (Random Forest)",
    "xgb": "부스팅 앙상블 (XGBoost)",
    "mlp": "딥러닝 (MLP, Base)",
    "cmlp": "딥러닝 (C-MLP, 비대칭 손실 α=1.50 · 논문 최종 모델)",
}


def build_mlp(n_in: int = 21, hidden: int = 256, dropout: float = 0.3):
    """논문 Table 4 최종 구조: Linear(hidden) → BatchNorm → ReLU → Dropout → Linear(1)."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class _DemoModel:
    key, name, kind = "demo", "휴리스틱 (연결 테스트 전용, 학습 모델 아님)", "demo"
    metrics = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 논문 Table 7 경향(80대·8 PS 기준 주문 800↓→약 66대, 1,000→약 70대, 1,200↑→약 73대)을 흉내 낸 비례식
        u = X["order_volume"].to_numpy() / (X["num_picking_stations"].to_numpy() * 125.0)
        ratio = np.clip(0.87 + 0.12 * (u - 1.0), 0.6, 1.0)
        return X["current_num_robots"].to_numpy() * ratio


class _SklearnModel:
    def __init__(self, key, entry, model_dir):
        import joblib

        self.key, self.kind = key, "sklearn"
        self.name = entry.get("name", MODEL_FAMILY.get(key, key))
        self.metrics = entry.get("metrics", {})
        self.pipe = joblib.load(os.path.join(model_dir, entry["file"]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipe.predict(X), dtype=float).ravel()


class _TorchModel:
    def __init__(self, key, entry, model_dir):
        import joblib
        import torch

        self.key, self.kind = key, "torch"
        self.name = entry.get("name", MODEL_FAMILY.get(key, key))
        self.metrics = entry.get("metrics", {})
        self.pre = joblib.load(os.path.join(model_dir, entry["preprocess"]))
        n_in = int(entry.get("n_features", 21))
        self.net = build_mlp(n_in, int(entry.get("hidden", 256)), float(entry.get("dropout", 0.3)))
        self.net.load_state_dict(torch.load(os.path.join(model_dir, entry["state"]), map_location="cpu"))
        self.net.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        x = self.pre.transform(X)
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        with torch.no_grad():
            return self.net(torch.tensor(x, dtype=torch.float32)).squeeze(-1).numpy().astype(float)


class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: dict[str, object] = {}
        self.meta: dict = {}
        path = os.path.join(model_dir, "registry.json")
        if os.path.exists(path):
            self.meta = json.load(open(path, encoding="utf-8"))
            for key, entry in self.meta.get("models", {}).items():
                try:
                    cls = _TorchModel if entry.get("type") == "torch" else _SklearnModel
                    self.models[key] = cls(key, entry, model_dir)
                except Exception as e:  # 한 모델 로딩 실패가 서버 전체를 막지 않도록
                    self.meta.setdefault("load_errors", {})[key] = str(e)
            self.mode = "trained" if self.models else "demo"
        else:
            self.mode = "demo"
        if self.mode == "demo":
            self.models = {"demo": _DemoModel()}
        self.default = self.meta.get("default", "cmlp") if self.mode == "trained" else "demo"
        if self.default not in self.models:
            self.default = next(iter(self.models))

    # ---------------------------------------------------------------- API
    def resolve(self, key: str | None) -> str:
        if key in (None, "", "default"):
            return self.default
        if key not in self.models:
            raise ValueError(f"model={key!r} 은 사용할 수 없음. 선택 가능: {list(self.models)}")
        return key

    def predict(self, key: str | None, X: pd.DataFrame) -> np.ndarray:
        return self.models[self.resolve(key)].predict(X)

    def describe(self) -> list[dict]:
        return [{"key": k, "name": m.name, "type": m.kind, "metrics": m.metrics,
                 "default": k == self.default} for k, m in self.models.items()]
