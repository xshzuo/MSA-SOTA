#!/usr/bin/env python3
"""Build the release package for the MSA-SOTA project.

The package contains the implementation, the experiment scripts, the result
files and the documents.  Three classes of leak are refused outright:

  * credentials (a password-looking literal assigned to a field, an sshpass
    argument, a private key block);
  * machine-specific absolute paths in code files (they would break the
    package on any other machine);
  * private network addresses.

The build fails rather than producing a package that contains any of them.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import zipfile

CODE_SUFFIXES = (".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache"}
SKIP_NAMES = {".DS_Store"}
# build artefacts that do not belong in a release
DROP_SUFFIX = (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz")

# the machine-specific bits that must not travel with the package
PRIVATE_IP = re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
ABS_HOME = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")
SSH_PASSWORD_ARG = re.compile(r"sshpass\s+-p\s+(?!\$|\")([^\s\"]+)")
PASSWORD_FIELD = re.compile(
    r"(?:pass(?:wd|word)|secret|token|api[_-]?key)\s*[=:]\s*[\"'][^\"']{4,}[\"']",
    re.IGNORECASE)
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
# placeholders that are explicitly allowed
ALLOWED_PLACEHOLDERS = ("<password>", "<user>", "<host>", "REDACTED", "xxxx")


def is_doc(path):
    return os.path.splitext(path)[1].lower() in (".md", ".txt", ".rst")


def sanitise(path, rel):
    """The text that will actually be shipped (and therefore scanned)."""
    try:
        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return ""
    if rel.endswith(".csv"):
        # the result files record the exact command line of the run, which
        # contains the experiment machine's absolute paths.  That is
        # provenance, not a secret, but it does not belong in a package:
        # the home directory is rewritten to $HOME.
        text = re.sub(r"/(?:home|Users)/[A-Za-z0-9_.-]+/", "$HOME/", text)
    return text


def scan_text(text, rel):
    """Return a list of leak descriptions for one file's text."""
    problems = []
    if not text:
        return problems
    doc = is_doc(rel)
    for i, line in enumerate(text.split("\n"), 1):
        for m in SSH_PASSWORD_ARG.finditer(line):
            problems.append(f"{rel}:{i}: sshpass with a literal password")
        for m in PRIVATE_KEY.finditer(line):
            problems.append(f"{rel}:{i}: a private key block")
        for m in PASSWORD_FIELD.finditer(line):
            frag = m.group(0)
            if not any(p in frag for p in ALLOWED_PLACEHOLDERS):
                problems.append(f"{rel}:{i}: a password-looking literal")
        if not doc:
            for m in ABS_HOME.finditer(line):
                problems.append(f"{rel}:{i}: machine-specific absolute path")
            for m in PRIVATE_IP.finditer(line):
                problems.append(f"{rel}:{i}: a private network address")
    return problems


def scan(path, rel):
    return scan_text(sanitise(path, rel), rel)


def collect(root, out, prefix=None):
    """Collect the files under root, keeping the top-level directory name."""
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_NAMES or f.endswith(".pyc"):
                continue
            if f.startswith("DBG") or f.startswith("preds_mosi_DBG"):
                continue          # debugging leftovers
            if f.endswith(DROP_SUFFIX):
                continue          # build artefacts
            p = os.path.join(base, f)
            rel = os.path.relpath(p, root)
            if prefix:
                rel = os.path.join(prefix, rel)
            out.append((p, rel))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default="dist")
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    keep = ["code", "scripts", "docs", "jobs", "results", "paper", "README.md"]
    files = []
    for k in keep:
        p = os.path.join(root, k)
        if not os.path.exists(p):
            continue
        if os.path.isfile(p):
            files.append((p, k))
        else:
            collect(p, files, prefix=k)

    problems = []
    for p, rel in files:
        problems += scan(p, rel)
    if problems:
        print("发布包守卫命中，构建中止：")
        for p in problems[:50]:
            print("  ", p)
        print(f"共 {len(problems)} 处")
        return 1

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    manifest = []
    for p, rel in files:
        dst = os.path.join(out, "MSA-SOTA", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rel.endswith(".csv"):
            with open(dst, "w") as fh:
                fh.write(sanitise(p, rel))
        else:
            shutil.copy2(p, dst)
        manifest.append((rel, sha256(dst), os.path.getsize(dst)))
    with open(os.path.join(out, "MSA-SOTA", "MANIFEST.txt"), "w") as fh:
        fh.write("# path  sha256  bytes\n")
        for rel, digest, size in sorted(manifest):
            fh.write(f"{rel}  {digest}  {size}\n")
    n = len(manifest)
    print(f"发布包已建立：{out}/MSA-SOTA，{n} 个文件，"
          f"守卫通过（凭据 / 绝对路径 / 私网地址）")

    if args.zip:
        zp = os.path.abspath(args.zip)
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel, _, _ in manifest:
                zf.write(os.path.join(out, "MSA-SOTA", rel), f"MSA-SOTA/{rel}")
            zf.write(os.path.join(out, "MSA-SOTA", "MANIFEST.txt"), "MSA-SOTA/MANIFEST.txt")
        print(f"压缩包：{zp}  ({os.path.getsize(zp)/2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
