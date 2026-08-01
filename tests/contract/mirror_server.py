#!/usr/bin/env python3
"""T2 契约层：官方 SDK 镜像对照法的响应回放端。

对每个 case：把 tests/fixtures/openai/<fixture>.sse 的原始 SSE 字节回放给
官方 openai SDK 客户端（base_url 指向本服务器），并把 SDK 发出的请求体
落盘 build/contract/golden/<case>.json。

SDK 能完整消费 = 响应侧合规（fixture 是官方可解析的合法流）。
golden 请求体 = 请求侧参考（SDK 按官方 schema 构造），供对比。

用法：
    mirror_server.py <fixture>.sse <case_name> [port]
    （由 run_mirror.py 以子进程方式调用，HTTP 服务为单 case 一次性）
"""
import http.server
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
GOLDEN = REPO / "build" / "contract" / "golden"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length)
        GOLDEN.mkdir(parents=True, exist_ok=True)
        (GOLDEN / f"{self.server.case_name}.json").write_bytes(body)

        fixtures = REPO / "tests" / "fixtures" / self.server.provider
        fixture = (fixtures / self.server.fixture_file).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(fixture)))
        self.end_headers()
        self.wfile.write(fixture)
        self.close_connection = True

    def log_message(self, *args):  # 静默
        pass


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, fixture_file, case_name, provider):
        super().__init__(addr, _Handler)
        self.fixture_file = fixture_file
        self.case_name = case_name
        self.provider = provider


def main() -> None:
    fixture_file = sys.argv[1]
    case_name = sys.argv[2]
    provider = sys.argv[3] if len(sys.argv) > 3 else "openai"
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    server = _Server(("127.0.0.1", port), fixture_file, case_name, provider)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
