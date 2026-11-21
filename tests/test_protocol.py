"""End-to-end MCP protocol tests against the real server process.

Covers both eras the server answers on stdio: MCP 2026-07-28 (no
handshake, protocol version and client capabilities in every request's
_meta) and the older initialize handshake. No hardware needed: only
list_devices and the approval gate are exercised.
"""

import json
import subprocess
import sys

import pytest

MODERN = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": MODERN,
    "io.modelcontextprotocol/clientCapabilities": {},
}


def _session(messages):
    """Send messages one at a time, collecting a reply for each request.

    stdin stays open until every reply has arrived: the server exits as
    soon as stdin closes and drops calls still in flight.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "buspirate_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
    )
    replies = {}
    try:
        for msg in messages:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            if "id" in msg:
                line = proc.stdout.readline()
                assert line, "server closed stdout"
                reply = json.loads(line)  # every stdout line must be JSON-RPC
                replies[reply["id"]] = reply
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    return replies


def _req(i, method, params=None):
    return {"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}


def _call(i, name, arguments, meta=META):
    params = {"name": name, "arguments": arguments}
    if meta is not None:
        params["_meta"] = meta
    return _req(i, "tools/call", params)


@pytest.fixture(scope="module")
def modern():
    return _session([
        _req(1, "server/discover", {"_meta": META}),
        _req(2, "tools/list", {"_meta": META}),
        _call(3, "no_such_tool", {}),
        _call(4, "set_voltage", {"voltage_v": 3.3, "current_limit_ma": 100}),
        _call(5, "list_devices", {}),
        _req(6, "tools/list", {"_meta": {
            "io.modelcontextprotocol/protocolVersion": "2099-01-01",
            "io.modelcontextprotocol/clientCapabilities": {},
        }}),
        _req(7, "tools/list", {}),
    ])


class TestModernEra:
    def test_discover(self, modern):
        result = modern[1]["result"]
        assert MODERN in result["supportedVersions"]
        assert result["resultType"] == "complete"
        assert "tools" in result["capabilities"]
        info = result["_meta"]["io.modelcontextprotocol/serverInfo"]
        assert info["name"] == "buspirate-mcp"

    def test_tools_list(self, modern):
        result = modern[2]["result"]
        assert result["resultType"] == "complete"
        assert result["cacheScope"] == "public" and result["ttlMs"] > 0
        tools = {t["name"]: t for t in result["tools"]}
        assert "set_voltage" in tools
        # no-parameter tools declare it
        assert tools["list_devices"]["inputSchema"]["additionalProperties"] is False
        assert tools["set_voltage"]["annotations"]["destructiveHint"] is True

    def test_unknown_tool_is_protocol_error(self, modern):
        assert modern[3]["error"]["code"] == -32602

    def test_approval_gate_is_tool_error(self, modern):
        result = modern[4]["result"]
        assert result["isError"] is True
        assert result["structuredContent"]["confirmation_required"] is True
        # serialized JSON also in a text block
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]

    def test_success_has_structured_content(self, modern):
        result = modern[5]["result"]
        assert result["isError"] is False
        assert "devices" in result["structuredContent"]

    def test_unsupported_version(self, modern):
        error = modern[6]["error"]
        assert error["code"] == -32022
        assert MODERN in error["data"]["supported"]

    def test_missing_meta_rejected(self, modern):
        assert modern[7]["error"]["code"] == -32602


class TestLegacyEra:
    """Clients still on the initialize handshake keep working."""

    def test_handshake_and_calls(self):
        replies = _session([
            _req(1, "initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            }),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            _req(2, "tools/list"),
            _call(3, "set_voltage", {"voltage_v": 3.3, "current_limit_ma": 100}, meta=None),
        ])
        assert replies[1]["result"]["protocolVersion"] == "2025-06-18"
        assert len(replies[2]["result"]["tools"]) > 0
        assert replies[3]["result"]["isError"] is True
