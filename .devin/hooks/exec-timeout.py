#!/usr/bin/env python3
"""PreToolUse hook for `exec`: wrap shell commands in `timeout` so a hung
or infinite-looping process is KILLED instead of left running.

The exec tool's own `timeout` parameter only bounds how long the agent
*waits* — an overrun backgrounds the command and the process keeps
running. This hook puts a real wall-clock fuse on the process itself.

Policy:
  - foreground command (no `timeout` arg):    900s fuse
  - `timeout: 0` (deliberately backgrounded): 3600s fuse
  - already `timeout`-wrapped, a lone shell-state builtin (`cd`, `export`,
    `source`, ...), or an interactive `tty` session: passed through.

The command is re-encoded as base64 and decoded inside `bash -c`, so no
quoting/escaping of the original text is needed.
"""
import base64
import json
import re
import sys

FOREGROUND_FUSE = 900
BACKGROUND_FUSE = 3600

# Builtins that mutate persistent-shell state and must run unwrapped.
STATE_BUILTINS = ("cd", "export", "source", ".", "alias", "unalias", "set",
                  "unset", "declare", "local", "readonly", "pushd", "popd",
                  "umask", "eval")

# A leading `timeout ...` (allowing VAR=x prefixes) means the caller already
# chose a fuse.
ALREADY_WRAPPED = re.compile(r"^\s*(\w+=\S+\s+)*timeout\s")


def main() -> None:
    data = json.load(sys.stdin)
    ti = data.get("tool_input") or {}
    cmd = ti.get("command") or ""
    if not cmd.strip():
        return

    stripped = cmd.strip()
    if ALREADY_WRAPPED.match(stripped) or ti.get("tty"):
        return

    # Lone state builtin (no compound operators) must mutate the real shell.
    first = stripped.split(None, 1)[0].rstrip(";")
    if first in STATE_BUILTINS and not re.search(r"[&|;]", stripped):
        return

    fuse = BACKGROUND_FUSE if ti.get("timeout") == 0 else FOREGROUND_FUSE
    b64 = base64.b64encode(cmd.encode()).decode()
    wrapped = (
        f"timeout --kill-after=10s {fuse}s "
        f'bash -c "$(echo {b64} | base64 -d)"'
    )
    json.dump(
        {"hookSpecificOutput": {"PreToolUse": {"updatedInput": {"command": wrapped}}}},
        sys.stdout,
    )


if __name__ == "__main__":
    main()
