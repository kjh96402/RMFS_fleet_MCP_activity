"""
client_demo.py — Python 으로 MCP 연결 확인 (Claude 없이, mcp SDK 의 Client 사용)
================================================================================
실행:  python client_demo.py
하는 일: 이 파일이 MCP 클라이언트(Host 역할)가 되어 server.py 를 stdio 로 띄우고 → 도구 목록 조회 → 도구 호출 → 결과 출력
       (Claude Desktop 이 내부에서 하는 일과 동일한 절차를 눈으로 확인)
입력:   data/Data_single_row.csv 의 1행 (12개 열 = predict_fleet_size 의 인자)
"""
import asyncio
import csv
import json
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "server.py"), "--stdio"], cwd=HERE)
INT_COLUMNS = ("ORDER_NUM", "SKU_COUNT_OPTIONS", "NUM_AISLES", "NUM_CROSS_AISLES", "NUM_PS", "OPTION", "MAX_ROBOTS", "NUM_PODS")


def load_row() -> dict:
    """Data_single_row.csv 1행 → 도구 인자 dict (숫자 열은 숫자로 변환)."""
    with open(os.path.join(HERE, "data", "Data_single_row.csv"), encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    row = {k.strip(): v.strip() for k, v in row.items()}
    for k in INT_COLUMNS:
        row[k] = int(float(row[k]))
    row["MAKESPAN_AT_MAX_ROBOTS"] = float(row["MAKESPAN_AT_MAX_ROBOTS"])
    return row


def data(r):
    return r.structured_content or json.loads(r.content[0].text)


async def main():
    row = load_row()
    async with Client(SERVER) as c:                                   # ① 서버 프로세스 실행 + 초기화(capabilities 교환)
        tools = await c.list_tools()                                  # ② tools/list
        print("[1] 연결 성공 — 도구:", [t.name for t in tools.tools])
        d = data(await c.call_tool("list_models", {}))                # ③ tools/call
        print(f"[2] list_models → mode={d['mode']} · 모델={[m['key'] for m in d['models']]}")
        if d["mode"] == "demo":
            print("    (학습 모델 없음 → 먼저 python run_local.py 또는 Claude 에서 train_models 실행)")
        d = data(await c.call_tool("predict_fleet_size", row))
        print(f"[3] predict_fleet_size(주문 {row['ORDER_NUM']}건·PS {row['NUM_PS']}·로봇 {row['MAX_ROBOTS']}대, {d['model']}) "
              f"→ 추천 {d['recommended_robots']}대 / 절감 {d['robot_savings']}대  warnings={d['warnings']}")
        d = data(await c.call_tool("compare_models", row))
        print("[4] compare_models →", [(r["model"], r["recommended_robots"]) for r in d["rows"]])
        print("OK — MCP 연결 정상. 다음 단계: python tools/make_config.py 로 Claude Desktop 설정 만들기")


if __name__ == "__main__":
    asyncio.run(main())
