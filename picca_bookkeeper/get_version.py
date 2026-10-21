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
    try:
        mod = __import__(package)
        src_dir = Path(mod.__file__).parent.parent
        toml = src_dir / "pyproject.toml"
        if toml.exists():
            match = re.search(
                r'^version\s*=\s*"([^"]+)"', toml.read_text(), re.MULTILINE
            )
            if match:
                return match.group(1)
        # dynamic version (e.g. setuptools-scm): ask git
        result = subprocess.run(
            ["git", "describe", "--tags"],
            cwd=src_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (ImportError, AttributeError):
        pass
    return "unknown"


if __name__ == "__main__":
    for pkg in sys.argv[1:]:
        print(get_version(pkg))
