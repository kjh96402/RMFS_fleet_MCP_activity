"""
tools/make_config.py — Claude Desktop 연동 설정(claude_desktop_config.json) 자동 탐색 · 저장
==========================================================================================
Windows 의 Claude Desktop 은 설치 방식에 따라 "실제로 읽는" 설정 파일 위치가 다르다.
  ① Microsoft Store(MSIX) 최신 빌드 : %LOCALAPPDATA%\\Claude-Data\\claude_<패키지ID>\\claude_desktop_config.json
  ② Microsoft Store(MSIX) 이전 빌드 : %LOCALAPPDATA%\\Packages\\Claude_<패키지ID>\\LocalCache\\Roaming\\Claude\\claude_desktop_config.json
  ③ 일반 설치판(.exe)               : %APPDATA%\\Claude\\claude_desktop_config.json
  (Store 빌드는 ③ 위치를 읽지 않으므로, ③ 에만 저장하면 rmfs-fleet 가 Claude 에 나타나지 않는다)
이 스크립트는 PC 에 실제로 존재하는 위치를 모두 찾아, 각 파일에 rmfs-fleet 항목만 병합 저장한다.

실행 (VS Code 터미널, 가상환경 활성화 상태에서):
  python tools/make_config.py                    → 설정 내용과 감지된 위치만 출력 (저장 안 함)
  python tools/make_config.py --write            → 감지된 모든 설정 파일에 병합 저장 (기존 설정 유지 · .bak 백업)
  python tools/make_config.py --write --path "C:\\...\\claude_desktop_config.json"   → 지정한 파일에만 저장
주의: 저장 전에 Claude Desktop 을 완전히 종료(작업 표시줄 트레이 아이콘 ▸ Quit)하세요.
      실행 중이면 앱이 설정 파일을 다시 덮어써 rmfs-fleet 항목이 사라질 수 있습니다.
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_NAME = "rmfs-fleet"
CFG_FILE = "claude_desktop_config.json"


def pick_python():
    """Claude Desktop 이 server.py 를 실행할 Python — mcp 패키지가 설치된 인터프리터를 고른다.
    지금 실행 중인 Python 에 mcp 가 없으면(다른 가상환경 활성화 등) 저장소의 .venv 를 쓴다."""
    import importlib.util
    if importlib.util.find_spec("mcp") is not None:
        return sys.executable, None
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    venv_py = os.path.join(ROOT, ".venv", *sub)
    if os.path.exists(venv_py):
        return venv_py, f"[안내] 현재 Python 에 mcp 패키지가 없어 저장소 가상환경을 사용합니다 → {venv_py}"
    return sys.executable, ("[주의] 현재 Python 에 mcp 패키지가 없습니다. 가상환경을 활성화하고 "
                            "pip install -r requirements.txt 를 실행한 뒤 다시 실행하세요.")


PYTHON, PY_NOTE = pick_python()
ENTRY = {"command": PYTHON, "args": [os.path.join(ROOT, "server.py"), "--stdio"]}
try:  # 한글·기호 출력 시 콘솔 인코딩 오류 방지
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def candidate_paths():
    """(설명, 설정 파일 경로) 목록 — 실제 PC 에 폴더가 있는 위치만, 우선순위 순."""
    system = platform.system()
    if system == "Darwin":
        return [("macOS", os.path.expanduser("~/Library/Application Support/Claude/" + CFG_FILE))]
    if system != "Windows":
        return [("Linux", os.path.expanduser("~/.config/Claude/" + CFG_FILE))]

    local = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    roaming = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    found = []
    # ① Microsoft Store 최신 빌드 — 데이터 폴더가 Claude-Data\claude_<패키지ID> 로 이동
    for d in sorted(glob.glob(os.path.join(local, "Claude-Data", "claude_*"))):
        if os.path.isdir(d):
            found.append(("① Microsoft Store 빌드 (Claude-Data)", os.path.join(d, CFG_FILE)))
    # ② Microsoft Store 이전 빌드 — MSIX 가상화로 %APPDATA% 가 패키지 폴더 안으로 리디렉션
    for d in sorted(glob.glob(os.path.join(local, "Packages", "Claude_*", "LocalCache", "Roaming", "Claude"))):
        if os.path.isdir(d):
            found.append(("② Microsoft Store 빌드 (MSIX 가상화 경로)", os.path.join(d, CFG_FILE)))
    # ③ 일반 설치판 — 폴더가 있거나, 위 두 위치가 하나도 없을 때
    std = os.path.join(roaming, "Claude", CFG_FILE)
    if os.path.isdir(os.path.dirname(std)) or not found:
        found.append(("③ 일반 설치판 (%APPDATA%\\Claude)", std))
    return found


def claude_running():
    """Windows 에서 Claude Desktop 실행 여부 (확인 실패 시 None).
    Claude Code CLI 도 claude.exe 이므로, 실행 파일 경로로 Desktop 앱만 구분한다
    (Store 빌드: ...\\WindowsApps\\Claude_... / 일반 설치판: ...\\AnthropicClaude\\...)."""
    if platform.system() != "Windows":
        return None
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-Process -Name claude -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }"]
        paths = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.lower().splitlines()
        return any(("windowsapps" in p and "claude_" in p) or "anthropicclaude" in p for p in paths)
    except Exception:
        return None


def load_json(path):
    """기존 설정 읽기. (데이터, 오류 메시지) 반환 — 파일이 없으면 ({}, None)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}, None
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        return (data if isinstance(data, dict) else {}), None
    except Exception as e:
        return None, f"JSON 형식 오류 ({e})"


def write_entry(path):
    """path 에 rmfs-fleet 항목만 병합 저장. 성공 여부와 메시지 반환."""
    data, err = load_json(path)
    if err:
        return False, f"건너뜀 — {err}. 파일을 직접 열어 고치거나 --path 로 다른 파일을 지정하세요."
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")                      # 백업
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_NAME] = ENTRY
    data["mcpServers"] = servers                               # 다른 키(preferences 등)는 그대로 유지
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return True, "저장 완료" + (" (기존 파일은 .bak 로 백업)" if os.path.exists(path + ".bak") else "")


def describe(path):
    if not os.path.exists(path):
        return "파일 없음 (저장 시 새로 생성)"
    t = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    data, err = load_json(path)
    if err:
        return f"수정 {t} · {err}"
    has = SERVER_NAME in (data.get("mcpServers") or {})
    return f"수정 {t} · rmfs-fleet {'등록됨' if has else '없음'}"


def main(argv):
    print("=== Claude Desktop 에 등록할 내용 (mcpServers 에 병합) ===")
    print(json.dumps({"mcpServers": {SERVER_NAME: ENTRY}}, ensure_ascii=False, indent=2))
    if PY_NOTE:
        print(PY_NOTE)

    if "--path" in argv:
        i = argv.index("--path")
        if i + 1 >= len(argv):
            print("--path 뒤에 설정 파일 경로를 적어 주세요.")
            return 1
        targets = [("지정 경로", os.path.abspath(argv[i + 1]))]
    else:
        targets = candidate_paths()

    print("\n=== 감지된 설정 파일 위치 ===")
    for label, p in targets:
        print(f"- {label}\n    {p}\n    → {describe(p)}")
    if len(targets) > 1:
        print("  (위치가 여러 곳이면 모두에 저장합니다 — 앱은 그중 한 곳만 읽으며, 나머지 항목은 영향 없음)")

    if "--write" not in argv:
        print("\n저장하려면:  python tools/make_config.py --write")
        return 0

    if claude_running():
        print("\n[주의] Claude Desktop 이 실행 중입니다. 트레이 아이콘 ▸ Quit 으로 완전히 종료한 뒤 이 명령을 다시 실행하세요.")
        print("       (실행 중에 저장하면 앱이 설정 파일을 덮어써 rmfs-fleet 항목이 사라질 수 있습니다)")
        if "--force" not in argv:
            return 1

    print("\n=== 저장 ===")
    ok_any = False
    for label, p in targets:
        ok, msg = write_entry(p)
        ok_any = ok_any or ok
        print(f"- {label}: {msg}\n    {'저장 위치' if ok else '대상 파일'}: {p}")
    if ok_any:
        print("\n다음: Claude Desktop 실행 → Settings ▸ Developer 에서 rmfs-fleet 상태가 running 인지 확인")
        print("      (Edit Config 버튼이 다른 파일을 열 수 있으니, 확인은 위에 출력된 경로 기준으로 하세요)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
