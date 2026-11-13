"""
run_local.py — Python 작동 확인 (MCP·Claude 없이 모델만 직접 실행)
=================================================================
실행:  python run_local.py
하는 일: ① data/train_data.csv 확인 → ② 6개 모델 학습(논문 설정, quick) → ③ data/Data_single_row.csv 의 운영 조건 추정 → 모델별 비교표 출력
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fleet.features import engineer, normalize_scenario, to_frame, validate  # noqa: E402
from fleet.registry import ModelRegistry  # noqa: E402
from fleet.train import dataset_info, train_all  # noqa: E402

if __name__ == "__main__":
    info = dataset_info()
    print(f"[1] 데이터셋: {info['source']} · {info['n_rows']:,}행 · 타깃(TARGET) {info['target_stats']}")
    print("[2] 모델 학습 (논문 4.2절 설정, quick 모드) ...")
    train_all(quick=True)
    REG = ModelRegistry(os.path.join(HERE, "models"))
    with open(os.path.join(HERE, "data", "Data_single_row.csv"), encoding="utf-8-sig", newline="") as f:
        s = normalize_scenario(next(csv.DictReader(f)))                 # 입력 = Data_single_row.csv 의 1행 (12개 열)
    X = to_frame([engineer(s)])
    print(f"[3] 운영 조건(Data_single_row.csv): {s}")
    print(f"    warnings={validate(s)}")
    print(f"    {'모델':6s} {'추천 대수':>8s} {'절감':>5s} {'R2':>7s} {'MAE':>6s}")
    for key in REG.models:
        y = float(REG.predict(key, X)[0]); rec = int(max(1, min(round(y), s["MAX_ROBOTS"])))
        m = REG.models[key].metrics
        print(f"    {key:6s} {rec:8d} {s['MAX_ROBOTS'] - rec:5d} {m.get('R2', 0):7.3f} {m.get('MAE', 0):6.2f}")
    print("OK — Python 환경 정상. 다음 단계: python client_demo.py (MCP 연결 확인) → Claude Desktop 연동")
