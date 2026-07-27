"""Scan the public artifact for credentials, local paths, and model weights."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


ALLOWED_EMAILS = {"li002504@umn.edu"}
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "build", "dist", "reproduced", "runs",
}
WEIGHT_EXTENSIONS = {
    ".bin", ".ckpt", ".gguf", ".h5", ".onnx", ".pb", ".pt", ".pth",
    ".safetensors",
}
BINARY_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".pdf", ".png", ".zip"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github-token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "github-pat": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "api-key": re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    "windows-user-path": re.compile(r"[A-Za-z]:\\Users\\"),
    "posix-user-path": re.compile(r"(?:/home|/Users)/[a-z][a-z0-9_-]+"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _files(root: Path):
    for directory, names, filenames in os.walk(root):
        names[:] = [name for name in names if name not in SKIP_DIRS]
        for filename in filenames:
            yield Path(directory) / filename


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    findings: list[tuple[str, str, str]] = []
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        if path.name == Path(__file__).name:
            continue
        if path.suffix.lower() in WEIGHT_EXTENSIONS:
            findings.append((relative, "model-weight", path.suffix))
            continue
        if path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for name, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                findings.append((relative, name, match.group(0)[:60]))
        for email in set(EMAIL.findall(text)):
            domain = email.rsplit("@", 1)[1].lower()
            if (email.lower() not in ALLOWED_EMAILS
                    and domain not in ALLOWED_EMAIL_DOMAINS):
                findings.append((relative, "non-public-email", email))

    print(f"[check-release] scanned root: {root}")
    for relative, kind, evidence in findings:
        print(f"  HIGH  {kind:20s} {relative} :: {evidence}")
    if findings:
        print(f"\n[check-release] FAIL: {len(findings)} finding(s)")
        return 1
    print("[check-release] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
