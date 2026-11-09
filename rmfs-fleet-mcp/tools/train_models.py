"""
tools/train_models.py — 터미널에서 모델군 학습 (server.py 의 train_models 도구와 동일한 코드)
사용:  python tools/train_models.py                      # data/dataset.csv (없으면 dataset_sample.csv) 전체 학습
       python tools/train_models.py --quick              # 빠른 점검
       python tools/train_models.py --models rf cmlp --alpha 2.0
       python tools/train_models.py --dataset 경로/다른데이터.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet.train import ALL_MODELS, train_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    train_all(a.dataset, a.models, {"alpha": a.alpha, "hidden": a.hidden, "dropout": a.dropout, "lr": a.lr}, quick=a.quick)
