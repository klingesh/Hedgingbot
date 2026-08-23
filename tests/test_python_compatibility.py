"""
Guard the minimum supported Python version.

Why this file exists
--------------------
`from __future__ import annotations` defers ANNOTATIONS, but not assignments. A
module-level type alias like

    Series = list[float | None]

is evaluated at import time, so it silently raises the whole package's minimum
version to Python 3.10 — and the failure only appears on the older interpreter,
with a message ("unsupported operand type(s) for |") that does not obviously point
at the cause.

That exact bug shipped in the first Phase 1 commit and was caught by running
preflight.py under Python 3.9. These tests make it impossible to reintroduce
without a red test, because CI here cannot run every interpreter.

Tradingbot's VPS may be on an older Python than a dev laptop, so this matters
practically and not just theoretically.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest

MIN_VERSION = (3, 9)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "scripts")


def python_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for d in SOURCE_DIRS:
        out.extend(sorted((ROOT / d).rglob("*.py")))
    return [p for p in out if "__pycache__" not in p.parts]


def test_there_are_source_files_to_check():
    """Guard against the glob silently matching nothing."""
    files = python_files()
    assert len(files) >= 15, f"only found {len(files)} source files; glob is wrong"


class _RuntimeUnionFinder(ast.NodeVisitor):
    """Find `X | Y` in positions Python evaluates at import time.

    Annotations are skipped because `from __future__ import annotations` turns
    them into strings. Everything else — module and class level assignments,
    default argument values, base classes — is evaluated for real.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    # Do not descend into annotations: they are deferred.
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node) -> None:
        # Default VALUES are evaluated at definition time; annotations are not.
        for default in list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]:
            self.visit(default)
        for stmt in node.body:
            self.visit(stmt)
        for dec in node.decorator_list:
            self.visit(dec)

    #: Builtin/typing names that appearing in `X | Y` means a TYPE union rather
    #: than arithmetic.
    TYPE_NAMES = frozenset({
        "int", "str", "float", "bool", "bytes", "complex", "object",
        "list", "dict", "set", "tuple", "frozenset", "bytearray",
        "Any", "Optional", "Union", "Callable", "Sequence", "Mapping",
        "Iterable", "Iterator", "Awaitable", "Coroutine", "Type",
    })

    def _looks_like_a_type(self, node: ast.expr) -> bool:
        """Is this operand plausibly a type rather than a number?

        The first version flagged EVERY BitOr, which made
        `fcntl.LOCK_EX | fcntl.LOCK_NB` — an ordinary bitwise OR of two ints —
        look like a Python 3.10 syntax problem. A checker that cries wolf on
        legitimate bit flags gets deleted, so it has to discriminate.

        `X | None` is the overwhelmingly common PEP 604 form and is decisive.
        Otherwise a bare builtin type name, or a subscripted generic like
        `list[int]`, is treated as a type.
        """
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Name) and node.id in self.TYPE_NAMES:
            return True
        if isinstance(node, ast.Subscript):
            return self._looks_like_a_type(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return (self._looks_like_a_type(node.left)
                    or self._looks_like_a_type(node.right))
        return False

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and (
            self._looks_like_a_type(node.left)
            or self._looks_like_a_type(node.right)
        ):
            self.hits.append((node.lineno, ast.unparse(node)
                              if hasattr(ast, "unparse") else "X | Y"))
        self.generic_visit(node)


def test_no_pep604_unions_are_evaluated_at_runtime():
    """`X | None` outside a deferred annotation would require Python 3.10.

    The canonical offender is a module-level alias. Use typing.Optional /
    typing.Union there instead.
    """
    offenders: list[str] = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        finder = _RuntimeUnionFinder()
        for stmt in tree.body:
            finder.visit(stmt)
        for lineno, snippet in finder.hits:
            offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {snippet}")

    assert not offenders, (
        "PEP 604 unions evaluated at runtime raise the minimum Python to 3.10.\n"
        "Use typing.Optional/Union in these positions instead:\n  "
        + "\n  ".join(offenders)
    )


def test_every_module_defers_its_annotations():
    """Any file using modern annotation syntax needs the future import.

    Files with no annotations at all (the `__init__.py` re-exports) are exempt.
    """
    missing: list[str] = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))

        has_future = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )
        if has_future:
            continue

        annotated = any(
            isinstance(n, (ast.AnnAssign,))
            or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (n.returns is not None
                     or any(a.annotation is not None
                            for a in list(n.args.args) + list(n.args.kwonlyargs))))
            for n in ast.walk(tree)
        )
        if annotated:
            missing.append(str(path.relative_to(ROOT)))

    assert not missing, (
        "these files use annotations without `from __future__ import annotations`, "
        "so the annotations are evaluated at import time:\n  " + "\n  ".join(missing)
    )


def test_no_syntax_newer_than_the_minimum_version():
    """Every source file must COMPILE under the declared minimum.

    ast.parse with feature_version rejects syntax newer than MIN_VERSION, which
    catches match statements, PEP 695 type parameters and similar before they
    reach a VPS running an older interpreter.
    """
    if sys.version_info < (3, 8):
        pytest.skip("feature_version needs Python 3.8+")

    offenders: list[str] = []
    for path in python_files():
        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=MIN_VERSION,
            )
        except SyntaxError as exc:
            offenders.append(
                f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}"
            )

    assert not offenders, (
        f"syntax newer than Python {MIN_VERSION[0]}.{MIN_VERSION[1]}:\n  "
        + "\n  ".join(offenders)
    )


def test_readme_and_requirements_agree_on_no_hard_dependencies():
    """The claim 'no third-party runtime dependencies' must stay true.

    Anything beyond pytest and PyYAML in requirements.txt would make it false, and
    PyYAML itself is optional (the config loader falls back).
    """
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    requirements = [
        line.split("#")[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    names = {r.split(">=")[0].split("==")[0].strip().lower()
             for r in requirements if r}

    assert names <= {"pytest", "pyyaml"}, (
        f"unexpected runtime dependency in requirements.txt: {names}. "
        "The risk maths is deliberately stdlib-only."
    )


def test_no_runtime_import_of_pandas_or_numpy():
    """The live path must not acquire a scientific-stack dependency by accident."""
    offenders: list[str] = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in ("pandas", "numpy", "scipy", "sklearn"):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: imports {m}"
                    )

    assert not offenders, (
        "the overlay must stay stdlib-only so it runs on a bare VPS and the "
        "hedge-sizing maths stays readable:\n  " + "\n  ".join(offenders)
    )



# ---------------------------------------------------------------------------
# The union checker must discriminate types from bit flags
# ---------------------------------------------------------------------------


def _union_hits(source: str) -> list[str]:
    tree = ast.parse(source)
    finder = _RuntimeUnionFinder()
    for stmt in tree.body:
        finder.visit(stmt)
    return [snippet for _lineno, snippet in finder.hits]


def test_checker_flags_a_real_pep604_type_alias():
    """The bug it exists for: a module-level alias evaluated at import time."""
    assert _union_hits("Series = list[float | None]")
    assert _union_hits("Thing = int | str")
    assert _union_hits("X: object = None\nY = str | None")


def test_checker_ignores_bitwise_or_of_flags():
    """fcntl.LOCK_EX | fcntl.LOCK_NB is arithmetic, not a type union.

    The first version flagged every BitOr and reported src/state/lock.py as a
    Python 3.10 problem. A checker that cries wolf on legitimate bit flags gets
    deleted, which would lose the real protection.
    """
    assert _union_hits("mode = fcntl.LOCK_EX | fcntl.LOCK_NB") == []
    assert _union_hits("flags = 0x01 | 0x02") == []
    assert _union_hits("mask = a | b") == []
    assert _union_hits("perm = stat.S_IRUSR | stat.S_IWUSR") == []


def test_checker_still_ignores_deferred_annotations():
    """Annotations are strings under `from __future__ import annotations`, so a
    union there is harmless and must not be flagged."""
    assert _union_hits("def f(x: int | None = None) -> str | None: ...") == []
    assert _union_hits("class C:\n    field: float | None") == []


def test_checker_catches_a_union_in_a_default_VALUE():
    """Default values ARE evaluated at definition time, unlike annotations."""
    assert _union_hits("def f(t=int | None): ...")
