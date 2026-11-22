"""Keep attention selection, native adapters and model ownership separate."""

import ast
from pathlib import Path


def test_attention_python_dependency_boundaries():
    root = Path(__file__).resolve().parents[2] / "astrai" / "extension"
    files = list((root / "backend" / "attention").glob("*.py"))
    files.extend((root / "kernel").glob("attention*.py"))
    for file in files:
        tree = ast.parse(file.read_text())
        kernel = file.parent.name == "kernel"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("astrai.model", "astrai.inference")), file
                if kernel:
                    assert not module.startswith("astrai.extension.backend"), file
                elif module == "astrai.extension.runtime.loader":
                    assert "get_module" not in {alias.name for alias in node.names}, (
                        file
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        ("astrai.model", "astrai.inference")
                    ), file
