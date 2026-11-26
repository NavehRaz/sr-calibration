"""L0 source tests (static; no imports of the package):

  * NEVER `fp_curvefit_seed(ndims=5)`: it stores ExtH = 1e-4, i.e. h ~ 1 -- not a hazard seed
    (PHASE_B_DESIGN.txt section 4). The hazard seed is `fp_curvefit_makeham`;
  * the adapter's top level imports the standard library and `srbench.contract` only (lazy imports);
  * the estimator has no global four-parameter list left;
  * the observation model's relabel is rc1's, verbatim.
"""
from __future__ import annotations

import ast
import os
import sys

import _jax2_testlib as T

RUNTIME = sorted(f for f in os.listdir(T.PKG) if f.endswith(".py"))


def _tree(path):
    return ast.parse(open(path).read(), filename=path)


def _call_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)


def seed_ndims_violations(src, filename="<src>"):
    """Calls to fp_curvefit_seed that could pass ndims != 4: an `ndims` keyword that is not the literal 4,
    a fifth positional argument (ndims is the 5th parameter), or **kwargs / *args (cannot be checked).
    Also any reference to fp_curvefit_seed that is not a direct call (getattr, aliasing)."""
    out = []
    tree = ast.parse(src, filename=filename)
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "fp_curvefit_seed":
            calls.add(id(node.func))
            if len(node.args) >= 5 or any(isinstance(a, ast.Starred) for a in node.args):
                out.append((filename, node.lineno, "positional ndims / *args"))
            for kw in node.keywords:
                if kw.arg is None:
                    out.append((filename, node.lineno, "**kwargs"))
                elif kw.arg == "ndims" and not (isinstance(kw.value, ast.Constant) and kw.value.value == 4):
                    out.append((filename, node.lineno, f"ndims={ast.unparse(kw.value)}"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in calls:
            nm = node.id if isinstance(node, ast.Name) else node.attr
            if nm == "fp_curvefit_seed" and not isinstance(getattr(node, "ctx", None), ast.Store):
                out.append((filename, node.lineno, "reference that is not a direct call"))
        if isinstance(node, ast.Constant) and node.value == "fp_curvefit_seed":
            out.append((filename, node.lineno, "string reference (getattr?)"))
    return out


def test_no_fp_curvefit_seed_with_ndims_5_in_the_package():
    bad = []
    for f in RUNTIME:
        p = os.path.join(T.PKG, f)
        bad += seed_ndims_violations(open(p).read(), f)
    assert bad == [], bad


def test_the_seed_checker_catches_every_form():
    """Positive controls: the checker must flag each way of passing ndims=5."""
    cases = {
        "fp_curvefit_seed(t, e, ndims=5)": 1,
        "SRK.fp_curvefit.fp_curvefit_seed(t, e, 1.0, True, 5)": 1,
        "fp_curvefit_seed(t, e, **kw)": 1,
        "fp_curvefit_seed(*a)": 1,
        "f = fp_curvefit_seed\nf(t, e, ndims=5)": 1,
        "getattr(FC, 'fp_curvefit_seed')(t, e, ndims=5)": 1,
        "fp_curvefit_seed(t, e, xc=1.0)": 0,
        "fp_curvefit_seed(t, e, xc=1.0, ndims=4)": 0,
        "fp_curvefit_makeham(t, e, ndims=5)": 0,
    }
    for src, n in cases.items():
        assert len(seed_ndims_violations(src)) >= n and (n or not seed_ndims_violations(src)), src


def test_adapter_top_level_imports_stdlib_and_contract_only():
    tree = _tree(os.path.join(T.PKG, "__init__.py"))
    allowed_stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad = []
    for node in tree.body:                        # module level only: function bodies import lazily
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed_stdlib:
                    bad.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not (mod.split(".")[0] in allowed_stdlib or mod == "srbench.contract"
                    or mod.startswith("srbench.contract.")):
                bad.append(mod)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    bad.append(f"conditional import at line {sub.lineno}")
    assert bad == [], bad


def test_observation_model_is_stdlib_only():
    tree = _tree(os.path.join(T.PKG, "observation_model.py"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_estimator_has_no_global_parameter_list():
    """No `PARAMS` name anywhere, and no literal 4-length parameter loops: everything is obj.params / NP."""
    tree = _tree(os.path.join(T.PKG, "estimator.py"))
    names = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "PARAMS"]
    assert names == [], f"`PARAMS` used at lines {names}"
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _call_name(n) in ("range", "empty", "zeros") and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == 4:
            bad.append((n.lineno, ast.unparse(n)))
        if isinstance(n, ast.List) and [getattr(e, "value", None) for e in n.elts] == [1, 2, 3]:
            bad.append((n.lineno, ast.unparse(n)))
    assert bad == [], bad


def _func_body_dump(path, name):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                body = body[1:]                   # drop the docstring
            return ast.dump(ast.Module(body=body, type_ignores=[])), [a.arg for a in node.args.args]
    raise LookupError(name)


def test_relabel_is_rc1_verbatim():
    new = _func_body_dump(os.path.join(T.PKG, "observation_model.py"), "censor_tie_relabel")
    old = _func_body_dump(os.path.join(T.FIXTURE, "code", "adapter.py"), "censor_tie_relabel")
    assert new == old
    tol_new = [n for n in _tree(os.path.join(T.PKG, "observation_model.py")).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_TIE_REL_TOL"]
    tol_old = [n for n in _tree(os.path.join(T.FIXTURE, "code", "adapter.py")).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_TIE_REL_TOL"]
    assert ast.dump(tol_new[0]) == ast.dump(tol_old[0])


def test_claim_mapping_is_rc1_verbatim():
    new = _func_body_dump(os.path.join(T.PKG, "__init__.py"), "claim_kwargs")
    old = _func_body_dump(os.path.join(T.FIXTURE, "code", "adapter.py"), "claim_kwargs")
    assert new == old
