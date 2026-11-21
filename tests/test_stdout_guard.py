"""stdout must carry only MCP messages, whatever else prints.

The BPIO2 SDK print()s on many paths and esptool runs as a child process.
The server relies on the mcp SDK's stdio_server() pointing fd 1 at stderr
while it serves (mcp >= 2); this pins that behaviour.
"""

import subprocess
import sys
import textwrap


def test_stray_output_goes_to_stderr():
    code = textwrap.dedent("""
        import os
        import anyio
        from mcp.server.stdio import stdio_server

        async def main():
            async with stdio_server() as (read_stream, write_stream):
                print("sdk noise", flush=True)    # Python print, like the SDK
                os.write(1, b"fd noise\\n")        # raw fd 1, like a C extension
                os.system("echo child noise")      # child inheriting fd 1
                # app.run() closes these when the client goes away
                await write_stream.aclose()
                await read_stream.aclose()

        anyio.run(main)
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], input="", capture_output=True,
        text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    for noise in ("sdk noise", "fd noise", "child noise"):
        assert noise in proc.stderr
