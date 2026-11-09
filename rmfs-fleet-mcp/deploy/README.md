# deploy/ — (선택) 원격 배포용 · 기본 실습에는 필요 없음

기본 실습은 학생 PC 에서 `server.py --stdio` 를 Claude Desktop 이 직접 실행하는 **로컬 방식**이라 서버 배포가 필요 없다.
서버를 한 곳에 띄워 여러 사람이 URL 로 쓰고 싶을 때만 아래를 사용한다.

- `render.yaml` — Render Blueprint. 사용하려면 **저장소 루트로 복사**한 뒤 Render ▸ New ▸ Blueprint ▸ 저장소 선택.
  배포 후 서버는 `https://<app>.onrender.com/mcp` (Streamable HTTP) 로 열린다. 무료 플랜은 15분 유휴 시 슬립.
- `smoke_test_http.py` — HTTP 모드 점검: `python server.py` (포트 8000) 실행 후 `python deploy/smoke_test_http.py [URL]`.
- 연결: Claude.ai ▸ Settings ▸ Connectors ▸ Add custom connector ▸ 위 URL (원격 MCP 는 유료 플랜 요건이 바뀔 수 있으니 support.claude.com 확인).
