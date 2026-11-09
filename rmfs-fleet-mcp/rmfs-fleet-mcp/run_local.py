"""
run_local.py — Python 작동 확인 (MCP·Claude 없이 모델만 직접 실행)
=================================================================
실행:  python run_local.py
하는 일: ① 데이터셋 확인 → ② 6개 모델 학습(논문 설정, quick) → ③ 대표 시나리오(주문 1,000건·PS 8·로봇 80대) 추정 → 모델별 비교표 출력
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet.features import engineer, normalize_scenario, to_frame, validate  # noqa: E402
from fleet.registry import ModelRegistry  # noqa: E402
from fleet.train import dataset_info, train_all  # noqa: E402

SCENARIO = dict(order_volume=1000, num_picking_stations=8, num_pods=2176, max_robots=80, makespan_at_max_robots=4400,
                skus_per_order=2, skus_per_pod=20, aisle_configuration="general", picking_station_placement="long_both",
                order_assignment="shared_sku_clustering", sku_pod_location="same_class_zoning", robot_assignment_flexibility="pooled")

if __name__ == "__main__":
    info = dataset_info()
    print(f"[1] 데이터셋: {info['source']} · {info['n_rows']:,}행 · 타깃(optimal_robots) {info['target_stats']}")
    print("[2] 모델 학습 (논문 4.2절 설정, quick 모드) ...")
    reg = train_all(quick=True)
    REG = ModelRegistry(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    s = normalize_scenario(SCENARIO)
    X = to_frame([engineer(s)])
    print(f"[3] 대표 시나리오: 주문 {s['order_volume']}건 · PS {s['num_picking_stations']}개 · 보유 로봇 {s['max_robots']}대  warnings={validate(s)}")
    print(f"    {'모델':6s} {'추천 대수':>8s} {'절감':>5s} {'R2':>7s} {'MAE':>6s}")
    for key in REG.models:
        y = float(REG.predict(key, X)[0]); rec = int(max(1, min(round(y), s["max_robots"])))
        m = REG.models[key].metrics
        print(f"    {key:6s} {rec:8d} {s['max_robots'] - rec:5d} {m.get('R2', 0):7.3f} {m.get('MAE', 0):6.2f}")
    print("OK — Python 환경 정상. 다음 단계: python client_demo.py (MCP 연결 확인) → Claude Desktop 연동")
