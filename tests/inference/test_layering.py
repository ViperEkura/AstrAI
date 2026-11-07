import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "astrai" / "inference"


class RuntimeImports(ast.NodeVisitor):
    def __init__(self, package):
        self.package = package
        self.imports = []

    def visit_If(self, node):
        test = node.test
        type_only = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if type_only:
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node):
        self.imports.extend((node.lineno, alias.name) for alias in node.names)

    def visit_ImportFrom(self, node):
        name = node.module or ""
        if node.level:
            name = resolve_name("." * node.level + name, self.package)
        self.imports.append((node.lineno, name))
        self.imports.extend(
            (node.lineno, f"{name}.{alias.name}") for alias in node.names
        )


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        ("worker", ("core", "frontend", "network")),
        ("core", ("frontend", "network")),
        ("contracts.py", ("core", "worker", "frontend", "network")),
    ],
)
def test_runtime_dependencies_follow_inference_layers(path, forbidden):
    target = INFERENCE_ROOT / path
    assert target.exists()
    files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
    forbidden = tuple(f"astrai.inference.{layer}" for layer in forbidden)
    violations = []
    for file in files:
        relative = file.relative_to(INFERENCE_ROOT)
        package = ".".join(("astrai", "inference", *relative.parts[:-1]))
        visitor = RuntimeImports(package)
        visitor.visit(ast.parse(file.read_text()))
        for line, module in visitor.imports:
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in forbidden
            ):
                violations.append(f"{relative}:{line}: {module}")
    assert not violations, "Runtime layer violations:\n" + "\n".join(violations)
