"""Regression test: Facecam must survive pythonw.exe.

The Start menu shortcut launches pythonw.exe (no console), where sys.stdout and
sys.stderr are None. CI runs python.exe, which always has a console, so the
existing selftest and e2e never see that state. This test recreates it,
imports facecam, then draws a tqdm bar the way the first-run model download
does. Before the fix that bar raised:

    AttributeError: 'NoneType' object has no attribute 'write'

Put it next to e2e.py and run it like e2e.py:

    .venv\\Scripts\\python.exe .github\\test_windowless_stdio.py [repo_dir]

repo_dir defaults to the folder above this file. Exits 0 on pass, 1 on fail.
Side effect: facecam's fix appends a few lines to <repo_dir>\\facecam.log.
"""
import os
import sys

report = sys.__stdout__  # the real console, kept for this test's own output
repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, repo)

sys.stdout = sys.stderr = None  # exactly what pythonw.exe hands a script

try:
    import facecam  # noqa: F401  the guard has to run at import, before the engine loads
    from tqdm import tqdm
    for _ in tqdm(range(3), desc="like the model download"):
        pass
    sys.stderr.write("stderr write ok\n")
except Exception as e:
    report.write(f"FAIL: {type(e).__name__}: {e}\n")
    sys.exit(1)
report.write("PASS: facecam runs with no stdout/stderr (pythonw.exe)\n")
