"""
연결 점검 스크립트 — 서버가 Tools/Resources/Prompts 를 제대로 노출하고 모든 도구가 동작하는지 확인
사용:  python deploy/smoke_test_http.py                                   (로컬 http://127.0.0.1:8000/mcp)
       python deploy/smoke_test_http.py https://xxx.onrender.com/mcp      (배포 URL)
"""
import asyncio
import json
import sys
import xml.dom.minidom

from mcp.client.client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"
BASE = dict(num_picking_stations=8, num_pods=2176, max_robots=80, makespan_at_max_robots=4400)


def data(r):
    return r.structured_content or json.loads(r.content[0].text)


async def main():
    async with Client(URL) as c:
        tools = await c.list_tools()
        print("[1] Tools     :", [t.name for t in tools.tools])
        print("[2] Resources :", [str(r.uri) for r in (await c.list_resources()).resources])
        print("[3] Prompts   :", [p.name for p in (await c.list_prompts()).prompts])

        d = data(await c.call_tool("list_models", {}))
        print(f"[4] list_models → mode={d['mode']} default={d['default']} models={[m['key'] for m in d['models']]}")
        if d.get("load_errors"):
            print("    load_errors:", d["load_errors"])

        d = data(await c.call_tool("predict_fleet_size", {"order_volume": 1000, **BASE}))
        print(f"[5] predict_fleet_size(1000건, {d['model']}) → {d['recommended_robots']}대 / 절감 {d['robot_savings']}대 / warnings={d['warnings']}")

        d = data(await c.call_tool("compare_models", {"order_volume": 1000, **BASE}))
        print("[6] compare_models :", [(r["model"], r["recommended_robots"]) for r in d["rows"]])

        d = data(await c.call_tool("whatif_order_volume", {"start": 700, "stop": 1300, "step": 100, **BASE}))
        print("[7] whatif         :", [(x["order_volume"], x["recommended_robots"]) for x in d["rows"]])

        tpl = (await c.read_resource("scenario://template")).contents[0].text
        d = data(await c.call_tool("run_experiment_from_csv", {"csv_text": tpl, "model": "all", "include_layout": True}))
        print(f"[8] run_experiment_from_csv → {d['summary']['n_ok']}/{d['summary']['n_rows']}행, 평균 절감 {d['summary']['avg_robot_savings']}대 "
              f"({d['summary']['avg_savings_rate_pct']}%), 센터별:",
              [(r["center_id"], r["predictions"][d['summary']['models'][0]]["recommended_robots"]) for r in d["results"] if "error" not in r])

        d = data(await c.call_tool("describe_layout", {"aisle_configuration": "general", "num_picking_stations": 8,
                                                        "picking_station_placement": "long_both", "num_pods": 2176, "max_robots": 80}))
        xml.dom.minidom.parseString(d["svg"])                      # SVG 가 올바른 XML 인지
        sp = d["spec"]
        print(f"[9] describe_layout → 통로 {sp['aisles']}×{sp['cross_aisles']}, 블록 {sp['block_grid']['blocks']}, 용량 {sp['pod_capacity']}, "
              f"점유율 {sp['storage_occupancy']}, PS edges={sp['station_edges']}, svg {len(d['svg']):,} bytes")

        d = data(await c.call_tool("predict_fleet_size", {"order_volume": 2000, **BASE}))
        print("[10] OOD(2000건) warnings:", d["warnings"][:1])
        p = await c.get_prompt("model_comparison_report", {"order_volume": "1000", "num_picking_stations": "8", "max_robots": "80"})
        print("[11] prompt:", p.messages[0].content.text[:50], "...")
        print("OK — 서버 정상")


asyncio.run(main())
