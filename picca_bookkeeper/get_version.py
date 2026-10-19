"""Get a package version, falling back to pyproject.toml for PYTHONPATH installs."""
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path


def get_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
    # Fast path: find versioned install dir in sys.path without importing the module
    # (importing can be expensive for heavy packages like vega)
    for entry in sys.path:
        p = Path(entry)
        dir_match = re.search(r"[-_](\d+\.\d+[\w.]*)$", p.name)
        if dir_match and p.name.lower().startswith(package.lower()):
            return dir_match.group(1)
    try:
        mod = __import__(package)
        if hasattr(mod, "__version__") and mod.__version__ and mod.__version__ != "unknown":
            return mod.__version__
        pkg_dir = Path(mod.__file__).parent
        toml = None
        for candidate in [pkg_dir.parent, pkg_dir.parent.parent]:
            if (candidate / "pyproject.toml").exists():
                toml = candidate / "pyproject.toml"
                break
        if toml is not None and toml.exists():
            match = re.search(
                r'^version\s*=\s*"([^"]+)"', toml.read_text(), re.MULTILINE
            )
            if match:
                return match.group(1)
        # dynamic version (e.g. setuptools-scm): ask git
        git_cwd = toml.parent if toml is not None else pkg_dir.parent
        result = subprocess.run(
            ["git", "describe", "--tags"],
            cwd=git_cwd,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        # Last resort: extract version from parent directory name (e.g. vega-1.8.0)
        dir_match = re.search(
            r"[-_](\d+\.\d+[\w.]*)$", pkg_dir.parent.name
        )
        if dir_match:
            return dir_match.group(1)
    except (ImportError, AttributeError, subprocess.TimeoutExpired):
        pass
    return "unknown"


if __name__ == "__main__":
    for pkg in sys.argv[1:]:
        print(get_version(pkg))
