#!/usr/bin/env python3
"""Every .github/workflows/*.yml must parse. GitHub reports an unparsable
workflow as a red run with ZERO jobs, which in the commit list is
indistinguishable from a red run with a failing leg -- and that is how CI ran
nothing for ~41 hours (a88e739..b4f9094, 2026-09-09/10). Milliseconds; first
step of `just gate-landing`."""
import glob, sys
import yaml
bad = 0
for f in sorted(glob.glob(".github/workflows/*.yml")):
    try:
        yaml.safe_load(open(f))
        print("ok     ", f)
    except Exception as e:  # noqa: BLE001
        bad += 1
        print("INVALID", f, "--", str(e).splitlines()[0])
sys.exit(1 if bad else 0)
