import io
import json
import os

from halflife.mcp import handle, serve

HERE = os.path.dirname(__file__)
AGENT = os.path.join(HERE, "..", "examples", "my_agent.py")


def call(name, arguments, mid=1):
    return handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": name, "arguments": arguments}})


def test_handshake_and_listing():
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
    assert {"run_benchmark", "replay_failures", "audit_traces", "bench_model"} <= set(names)
    assert handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"] == -32601


def test_fix_and_replay_loop(tmp_path):
    out = str(tmp_path)
    weak = tmp_path / "weak_agent.py"
    weak.write_text(
        "from halflife import benchmark\n"
        "from halflife.demo import ClaimsDemoAgent\n"
        "@benchmark(suite='claims-intake')\n"
        "def weak(task):\n"
        "    return ClaimsDemoAgent(base_error=0.05, fatigue=0.02, seed=7).run(task)\n"
    )
    res = call("run_benchmark", {"target": str(weak), "lengths": "5,10,20", "tasks": 2, "runs": 2, "out": out})["result"]
    assert not res["isError"], res
    text = res["content"][0]["text"]
    assert "Reliability half-life" in text and "evals:" in text
    pack = next(os.path.join(out, f) for f in os.listdir(out) if f.endswith(".evals.json"))

    fixed = tmp_path / "fixed_agent.py"
    fixed.write_text(
        "from halflife import benchmark\n"
        "from halflife.demo import ClaimsDemoAgent\n"
        "@benchmark(suite='claims-intake')\n"
        "def fixed(task):\n"
        "    return ClaimsDemoAgent(base_error=0, fatigue=0).run(task)\n"
    )
    res = call("replay_failures", {"target": str(fixed), "pack": pack, "runs": 1, "out": out})["result"]
    assert not res["isError"], res
    assert "Now passing: 100%" in res["content"][0]["text"]


def test_errors_are_returned_not_raised():
    res = call("run_benchmark", {"target": "/no/such/agent.py"})["result"]
    assert res["isError"] and "no such file" in res["content"][0]["text"]


def test_stdio_framing():
    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\nnot json\n')
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert lines[0]["result"] == {} and lines[1]["error"]["code"] == -32700
