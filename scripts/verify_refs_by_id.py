#!/usr/bin/env python3
"""Verify the remaining references by direct identifier.

Searching for these five by title is unreliable: the layered learning rate
decay paper and the MMSA toolkit paper are not indexed by Crossref or by the
arXiv title search, and the CMU-MOSI paper is a journal article.  Each one is
therefore looked up by the identifier that the work itself uses, and the title
of the returned record has to agree with the title the manuscript uses.
"""
import difflib
import json
import re
import subprocess
import urllib.parse

UA = {"User-Agent": "msa-sota/1.0 (mailto:xsh.zuo@hotmail.com)"}


def curl(url):
    cmd = ["curl", "-sL", "--max-time", "35"]
    for k, v in UA.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def norm(s):
    s = (s or "").replace("{", "").replace("}", "").replace("\\", "").replace("&", "and")
    s = re.sub(r"[^\w\s-]", " ", s.lower())
    return " ".join(s.split())


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# key -> (kind, identifier, expected title)
CHECKS = [
    ("llrd", "arxiv", "2010.05738",
     "Layer-wise Learning Rate Decay for Large Pretrained Models"),
    ("mmsa", "anthology", "2022.acl-demo.20",
     "M-SENA: An Integrated Platform for Multimodal Sentiment Analysis"),
    ("mosi", "crossref", "10.1109/MIS.2016.94",
     "Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages"),
    ("mmml", "arxiv", "2308.00264",
     "Multimodal Multi-loss Fusion Network for Sentiment Analysis"),
    ("mosei", "anthology", "P18-1208",
     "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph"),
    ("bert", "anthology", "N19-1423",
     "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
]

for key, kind, ident, want in CHECKS:
    if kind == "arxiv":
        xml = curl(f"https://export.arxiv.org/api/query?id_list={ident}")
        ents = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        if not ents:
            print(f"[无记录] {key}: arXiv {ident} 无条目")
            continue
        t = re.search(r"<title>(.*?)</title>", ents[0], re.S)
        title = " ".join(t.group(1).split()) if t else ""
        s = sim(want, title)
        print(f"[{'匹配' if s >= 0.8 else '不符'}] {key:7s} arXiv:{ident}  相似度 {s:.2f}")
        print(f"          {title[:96]}")
    elif kind == "anthology":
        raw = curl(f"https://aclanthology.org/{ident}.bib")
        t = re.search(r'title\s*=\s*"([^"]+)"', raw)
        if not t:
            print(f"[无记录] {key}: anthology {ident} 无 bib")
            continue
        title = t.group(1)
        s = sim(want, title)
        a = re.search(r'author\s*=\s*"([^"]+)"', raw)
        b = re.search(r'booktitle\s*=\s*"([^"]+)"', raw)
        y = re.search(r'year\s*=\s*"?(\d{4})', raw)
        d = re.search(r'doi\s*=\s*"([^"]+)"', raw)
        print(f"[{'匹配' if s >= 0.8 else '不符'}] {key:7s} anthology:{ident}  相似度 {s:.2f}")
        print(f"          {title[:96]}")
        print(f"          {re.sub(r'\\s+', ' ', a.group(1))[:90] if a else ''}")
        print(f"          {(b.group(1)[:70] if b else '')} | {y.group(1) if y else ''} | {d.group(1) if d else '-'}")
    else:
        raw = curl("https://api.crossref.org/works/" + urllib.parse.quote(ident))
        try:
            m = json.loads(raw)["message"]
        except Exception:
            print(f"[无记录] {key}: Crossref {ident} 无记录")
            continue
        title = (m.get("title") or [""])[0]
        s = sim(want, title)
        ct = (m.get("container-title") or [""])[0]
        yr = (m.get("issued", {}).get("date-parts") or [[""]])[0][0]
        au = ", ".join(a.get("family", "") for a in (m.get("author") or [])[:4])
        print(f"[{'匹配' if s >= 0.8 else '不符'}] {key:7s} {ident}  相似度 {s:.2f}")
        print(f"          {title[:96]}")
        print(f"          {ct[:60]} | {yr} | {au}")
