"""The FP path never imports numba (tasks T061; FR-014; research R13): a fresh process imports sraging and
sraging.engine and evaluates an FP likelihood, and numba is not in sys.modules; a static scan refuses a numba import
outside src/sraging/langevin/ (a planted file beside its twin)."""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(REPO, "src", "sraging")
_IMPORT = re.compile(r"^\s*(import\s+numba\b|from\s+numba\b)", re.M)


def numba_imports_outside_langevin(root):
    """Every .py file under `root` importing numba, outside a `langevin` directory."""
    bad = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        if os.path.basename(d) == "langevin" or "/langevin/" in d.replace(os.sep, "/") + "/":
            continue
        for f in files:
            if f.endswith(".py") and _IMPORT.search(open(os.path.join(d, f), encoding="utf-8").read()):
                bad.append(os.path.relpath(os.path.join(d, f), root))
    return bad


def test_an_fp_evaluation_leaves_numba_unimported():
    code = ("import sys, numpy as np\n"
            "import sraging, sraging.engine\n"
            "from sraging import Cohort, SRParams\n"
            "from sraging.engine import evaluator, Heterogeneity\n"
            "c = Cohort.from_arrays(np.arange(20.0, 120.0), np.ones(100, int), delta=1.0)\n"
            "v = evaluator(c, heterogeneity=Heterogeneity(parameter='xc', sd=0.2, M=5, check=False)).value("
            "SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1))\n"
            "import sraging.langevin\n"                                    # its package import alone is lazy too
            "print(np.isfinite(v), 'numba' in sys.modules)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=600,
                       env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-2:] == ["True", "False"], r.stdout


def test_the_static_scan_finds_no_numba_import_outside_langevin():
    assert numba_imports_outside_langevin(SRC) == []


def test_a_planted_numba_import_is_found_and_its_twin_inside_langevin_is_not(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "langevin").mkdir()
    (tmp_path / "engine" / "bad.py").write_text("import numpy\nfrom numba import njit\n")
    (tmp_path / "langevin" / "ok.py").write_text("import numba\n")
    assert numba_imports_outside_langevin(str(tmp_path)) == [os.path.join("engine", "bad.py")]
