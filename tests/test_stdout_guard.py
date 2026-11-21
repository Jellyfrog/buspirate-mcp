"""stdout must carry only MCP messages, whatever else prints."""

import subprocess
import sys
import textwrap


def test_stray_output_goes_to_stderr():
    code = textwrap.dedent("""
        import os
        from buspirate_mcp.server import claim_stdout_for_protocol
        out = claim_stdout_for_protocol()
        print("sdk noise")                      # Python print, like the SDK
        os.write(1, b"fd noise\\n")              # raw fd 1, like a C extension
        os.system("echo child noise")           # child process inheriting fd 1
        out.write("PROTOCOL\\n")
        out.flush()
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "PROTOCOL\n"
    for noise in ("sdk noise", "fd noise", "child noise"):
        assert noise in proc.stderr
