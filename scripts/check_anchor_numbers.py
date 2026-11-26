#!/usr/bin/env python3
"""Check that the anchor numbers really appear in the cited papers.

The manuscript compares its own readings against published numbers.  Each
anchor must be traceable to a specific table of a specific paper, so the
numbers are looked up in the text of the published PDF.
"""
import os
import re
import subprocess
import sys

CACHE = "/tmp/refpdf"
os.makedirs(CACHE, exist_ok=True)

# key -> (url, numbers that must appear in the text)
TARGETS = [
    ("mmml",    "https://aclanthology.org/2024.naacl-long.403.pdf",
     ["85.91", "88.16", "48.25", "0.6429"]),
    ("dpdflq",  "https://aclanthology.org/2025.emnlp-main.571.pdf",
     ["86.15", "88.26", "48.63", "0.638"]),
]


def download(url):
    name = url.rsplit("/", 1)[-1]
    path = os.path.join(CACHE, name)
    if not os.path.exists(path) or os.path.getsize(path) < 10000:
        subprocess.run(["curl", "-sL", "--max-time", "90", "-o", path, url],
                       capture_output=True)
    return path


PDFTEXTSWIFT = "/tmp/pdftext"
PDFTEXTSRC = os.path.expanduser("~/.hermes/cache/scratch/pdftext.swift")


def text_of(path):
    """Extract the text of a PDF with the local PDFKit helper.

    The helper has to be compiled: running the script in the interpreter
    fails to load the PDFKit symbols.
    """
    if not os.path.exists(PDFTEXTSWIFT):
        subprocess.run(["swiftc", "-O", PDFTEXTSRC, "-o", PDFTEXTSWIFT],
                       capture_output=True)
    out = subprocess.run([PDFTEXTSWIFT, path], capture_output=True, text=True)
    return out.stdout


for key, url, numbers in TARGETS:
    path = download(url)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"\n=== {key}  {url}")
    if size < 10000:
        print(f"    PDF 取不到（{size} B）")
        continue
    txt = text_of(path)
    print(f"    {size} B, 文本 {len(txt)} 字符")
    for n in numbers:
        hits = len(re.findall(re.escape(n), txt))
        # also accept the number written without the trailing zero
        alt = n.rstrip("0").rstrip(".")
        alt_hits = len(re.findall(re.escape(alt) + r"(?!\d)", txt))
        print(f"    {n:8s} 出现 {hits:3d} 次" + (f"（含 {alt} 形式 {alt_hits} 次）" if alt != n else ""))
