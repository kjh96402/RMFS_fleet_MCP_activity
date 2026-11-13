"""
layout.py — RMFS 레이아웃 개략도 (논문 Table 1 통로 구성 기준) 스펙 계산 + SVG 생성
===================================================================================
통로 구성(NUM_AISLES × NUM_CROSS_AISLES) : Square 15×15 (종횡비 1.0) · General 12×18 (1.5) · Elongated 9×27 (3.0)
보관 블록                          : (aisle+1) × (cross-aisle+1) 개, 블록당 Pod 10개 (2 m × 5 m, Pod 1 m × 1 m)
Pod 수                             : 블록 수 × 10 × 점유율(0.8~1.0)  → 입력 num_pods 로 점유율 역산
피킹 스테이션 배치(PLACEMENT_CODE) : BL · BS · OL · OS = Both/One side × Long/Short side  (긴 변 = 블록 행이 많은 세로변)
"""
from __future__ import annotations

from fleet.features import aisle_configuration as _aisle_configuration

PODS_PER_BLOCK = 10


def layout_spec(num_aisles: int, num_cross_aisles: int, num_picking_stations: int, placement_code: str,
                num_pods: int | None = None, max_robots: int | None = None) -> dict:
    n_a, n_c = int(num_aisles), int(num_cross_aisles)
    ratio = round(n_c / n_a, 2)
    aisle_configuration = str(_aisle_configuration(n_a, n_c))
    cols, rows = n_a + 1, n_c + 1
    blocks = cols * rows
    capacity = blocks * PODS_PER_BLOCK
    pods = num_pods if num_pods else capacity
    occupancy = round(min(pods / capacity, 1.0), 3)
    long_is_vertical = rows >= cols
    both, side = ("both" if placement_code[0] == "B" else "one"), ("long" if placement_code[1] == "L" else "short")
    if (side == "long") == long_is_vertical:
        edges = ["left", "right"] if both == "both" else ["left"]
    else:
        edges = ["top", "bottom"] if both == "both" else ["top"]
    # 스테이션을 선택된 변에 균등 분배
    stations = []
    per_edge = [num_picking_stations // len(edges) + (1 if i < num_picking_stations % len(edges) else 0) for i in range(len(edges))]
    k = 1
    for edge, n in zip(edges, per_edge):
        for j in range(n):
            stations.append({"id": f"PS{k}", "edge": edge, "pos": round((j + 1) / (n + 1), 3)})
            k += 1
    return {
        "aisle_configuration": aisle_configuration, "aisles": n_a, "cross_aisles": n_c, "aspect_ratio": ratio,
        "block_grid": {"cols": cols, "rows": rows, "blocks": blocks, "pods_per_block": PODS_PER_BLOCK},
        "pod_capacity": capacity, "num_pods": pods, "storage_occupancy": occupancy,
        "placement_code": placement_code, "station_edges": edges, "stations": stations,
        "max_robots": max_robots, "length_m": rows * 5 + n_c, "width_m": cols * 2 + n_a,
    }


def layout_svg(spec: dict, title: str = "RMFS 레이아웃 개략도", width_px: int = 640) -> str:
    cols, rows = spec["block_grid"]["cols"], spec["block_grid"]["rows"]
    margin, cell, gap = 46, 1.0, 0.35            # 블록 1칸 = 1.0, 통로 폭 = 0.35 (상대 단위)
    gw, gh = cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap
    scale = (width_px - 2 * margin) / gw
    H = int(gh * scale + 2 * margin + 60)
    W = width_px
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial, sans-serif">',
         '<defs><pattern id="pod" width="1" height="1" patternUnits="objectBoundingBox" patternContentUnits="objectBoundingBox">'
         '<rect width="1" height="1" fill="#DCE6F5"/>'
         + "".join(f'<rect x="{i/5+0.03}" y="{j/2+0.06}" width="0.14" height="0.38" fill="#3B6FD6"/>' for i in range(5) for j in range(2))
         + '</pattern></defs>',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
         f'<text x="{margin}" y="24" font-size="15" font-weight="bold" fill="#1F3A6E">{title}</text>']
    ox, oy = margin, 40
    for r in range(rows):
        for c in range(cols):
            x, y = ox + c * (cell + gap) * scale, oy + r * (cell + gap) * scale
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell*scale:.1f}" height="{cell*scale:.1f}" fill="url(#pod)" stroke="#9FB4DB" stroke-width="0.5"/>')
    # 피킹 스테이션
    gx1, gy1 = ox + gw * scale, oy + gh * scale
    for st in spec["stations"]:
        e, p, s = st["edge"], st["pos"], 14
        if e in ("left", "right"):
            x = (ox - s - 8) if e == "left" else (gx1 + 8); y = oy + p * gh * scale - s / 2
        else:
            y = (oy - s - 8) if e == "top" else (gy1 + 8); x = ox + p * gw * scale - s / 2
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" fill="#E0301E" rx="2"/>')
        tx = x + s / 2
        ty = y - 3 if e != "bottom" else y + s + 11
        if e in ("left", "right"):
            ty = y + s + 11
        o.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="9" text-anchor="middle" fill="#333">{st["id"]}</text>')
    # 범례 및 파라미터
    ly = int(gy1) + 38
    o.append(f'<rect x="{margin}" y="{ly-10}" width="12" height="12" fill="url(#pod)" stroke="#9FB4DB"/>'
             f'<text x="{margin+16}" y="{ly}" font-size="10" fill="#333">보관 블록 (Pod 10개)</text>'
             f'<rect x="{margin+150}" y="{ly-10}" width="12" height="12" fill="#E0301E"/>'
             f'<text x="{margin+166}" y="{ly}" font-size="10" fill="#333">피킹 스테이션</text>')
    info = (f'통로 {spec["aisles"]} × 교차통로 {spec["cross_aisles"]} ({spec["aisle_configuration"]}, 종횡비 {spec["aspect_ratio"]}) · '
            f'블록 {spec["block_grid"]["blocks"]} · Pod {spec["num_pods"]:,} / 용량 {spec["pod_capacity"]:,} (점유율 {spec["storage_occupancy"]:.0%}) · '
            f'PS {len(spec["stations"])}개 ({spec["placement_code"]})' + (f' · 보유 로봇 {spec["max_robots"]}대' if spec.get("max_robots") else ""))
    o.append(f'<text x="{margin}" y="{ly+18}" font-size="9.5" fill="#555">{info}</text>')
    o.append("</svg>")
    return "\n".join(o)
