"""
tools/make_config.py — Claude Desktop 연동 설정(claude_desktop_config.json) 자동 생성
====================================================================================
실행:  python tools/make_config.py            → 화면에 설정 JSON 출력 (복사 → Claude Desktop ▸ Settings ▸ Developer ▸ Edit Config 에 붙여넣기)
       python tools/make_config.py --write    → Claude Desktop 설정 파일에 직접 병합 저장 (기존 서버 설정은 유지)
현재 사용 중인 python(가상환경 포함)과 server.py 의 절대 경로를 자동으로 넣어 주므로 경로 오타가 생기지 않는다.
"""
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
entry = {"command": sys.executable, "args": [os.path.join(ROOT, "server.py"), "--stdio"]}
config = {"mcpServers": {"rmfs-fleet": entry}}

if platform.system() == "Windows":
    cfg_path = os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude_desktop_config.json")
elif platform.system() == "Darwin":
    cfg_path = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
else:
    cfg_path = os.path.expanduser("~/.config/Claude/claude_desktop_config.json")

print("=== 아래 내용을 Claude Desktop ▸ Settings ▸ Developer ▸ Edit Config 에 붙여넣기 ===")
print(json.dumps(config, ensure_ascii=False, indent=2))
print(f"=== 설정 파일 위치: {cfg_path}")

if "--write" in sys.argv:
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    existing = {}
    if os.path.exists(cfg_path):
        try:
            existing = json.load(open(cfg_path, encoding="utf-8"))
        except Exception:
            existing = {}
    existing.setdefault("mcpServers", {})["rmfs-fleet"] = entry
    json.dump(existing, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장 완료 → {cfg_path}\n이제 Claude Desktop 을 완전히 종료(트레이 아이콘 Quit)한 뒤 다시 실행하면 도구 아이콘에 rmfs-fleet 가 표시됩니다.")
else:
    print("(직접 저장하려면:  python tools/make_config.py --write)")
