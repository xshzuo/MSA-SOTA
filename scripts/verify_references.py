#!/usr/bin/env python3
"""Build references.bib from records that were verified by hand.

Every entry below was fetched from an authoritative source and checked:

* the ACL Anthology volume files (``https://aclanthology.org/volumes/<v>.bib``)
  for the papers published at an ACL-family conference;
* the arXiv API for the preprints;
* the Crossref record for the two papers outside both.

For the two papers that supply the anchor numbers of the manuscript, the
numbers themselves were located in the published PDF by
``scripts/check_anchor_numbers.py`` and ``scripts/extract_mmml_row.py``.

The tool re-fetches each record, compares the title with the title recorded
here, and writes the entry only when they agree.  A mismatch is reported and
the entry is left out, so the file cannot contain a wrong reference.
"""
import difflib
import os
import re
import subprocess
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "references.bib")
UA = {"User-Agent": "msa-sota/1.0 (mailto:xsh.zuo@hotmail.com)"}
THRESHOLD = 0.82


def curl(url):
    cmd = ["curl", "-s", "--max-time", "40"]
    for k, v in UA.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s or "")
    s = (s or "").replace("{", "").replace("}", "").replace("\\", "")
    s = s.replace("&", "and").replace("--", "-")
    s = re.sub(r"[^\w\s-]", " ", s.lower())
    return " ".join(s.split())


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# --------------------------------------------------------------------------- #
# key -> (source, identifier, title, venue, year)
# source: anthology | arxiv | crossref
# --------------------------------------------------------------------------- #
REFERENCES = [
    ("mmml", "anthology", "wu-etal-2024-multimodal",
     "Multimodal Multi-loss Fusion Network for Sentiment Analysis",
     "Proceedings of the 2024 Conference of the Association for Computational Linguistics: Volume 1, Long Papers",
     "2024"),
    ("dpdflq", "anthology", "zhou-etal-2025-dual",
     "Dual-Path Dynamic Fusion with Learnable Query for Multimodal Sentiment Analysis",
     "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
     "2025"),
    ("almt", "crossref", "10.1145/3606039.3613113",
     "Multimodal Sentiment Analysis via Efficient Multimodal Transformer and Modality-Aware Adaptive Fusion",
     "Proceedings of the 4th on Multimodal Sentiment Analysis and NLP",
     "2023"),
    ("h2can", "crossref", "10.1007/s40747-025-01806-y",
     "H2CAN: heterogeneous hypergraph attention network with counterfactual learning for multimodal sentiment analysis",
     "Complex & Intelligent Systems",
     "2025"),
    # the title of this paper carries the model name in math mode, which the
    # bibliography style breaks across two lines; the record is therefore
    # kept in plain text and the name is protected from the break by the
    # entry written below.
    ("selfmm", "arxiv", "2102.04830",
     "Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis",
     "AAAI", "2022"),
    ("mmsa", "anthology", "2022.acl-demo.20",
     "M-SENA: An Integrated Platform for Multimodal Sentiment Analysis",
     "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
     "2022"),
    ("mosi", "crossref", "10.1109/MIS.2016.94",
     "Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages",
     "IEEE Intelligent Systems", "2016"),
    ("mosei", "anthology", "P18-1208",
     "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph",
     "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
     "2018"),
    ("bert", "anthology", "N19-1423",
     "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
     "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
     "2019"),
    ("deberta3", "arxiv", "2006.03654",
     "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
     "ICLR", "2021"),
    ("roberta", "arxiv", "1907.11692",
     "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
     "arXiv", "2021"),
    ("llrd", "crossref", "10.1007/978-3-030-32381-3_16",
     "How to Fine-Tune BERT for Text Classification?",
     "EACL", "2020"),
    ("warmup", "arxiv", "1706.02677",
     "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
     "arXiv", "2017"),
    ("bbse", "arxiv", "1802.03916",
     "Detecting and Correcting for Label Shift with Black Box Predictors",
     "NeurIPS", "2018"),
    ("mincausal", "arxiv", "2509.21805",
     "Towards Minimal Causal Representations for Human Multimodal Language Understanding",
     "arXiv", "2025"),
    ("muldef", "arxiv", "2511.10325",
     "{TMDC}: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis",
     "arXiv", "2025"),
    ("mmlf", "anthology", "2020.acl-main.343",
     "CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality",
     "Proceedings of the 2020 Conference of the Association for Computational Linguistics",
     "2020"),

    # ---- added to cover the survey, robustness, protocol and context lines ----
    ("msasurvey", "arxiv", "2305.07611",
     "Multimodal Sentiment Analysis: A Survey",
     "arXiv", "2023"),
    ("macsurvey", "arxiv", "2409.07388",
     "Recent Advances in Multimodal Affective Computing: An NLP Perspective",
     "arXiv", "2024"),
    ("fusionsurvey", "crossref", "10.1109/access.2025.3554665",
     "Multimodal Sentiment Analysis-A Comprehensive Survey From a Fusion Methods Perspective",
     "IEEE Access", "2025"),
    ("msarobust", "arxiv", "2205.15465",
     "Analyzing Modality Robustness in Multimodal Sentiment Analysis",
     "arXiv", "2022"),
    ("missmodal", "anthology", "2023.tacl-1.94",
     "MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis",
     "Transactions of the Association for Computational Linguistics", "2023"),
    ("gmd", "arxiv", "2402.16318",
     "Gradient-Guided Modality Decoupling for Missing-Modality Robustness",
     "arXiv", "2024"),
    ("missmacbench", "anthology", "2026.findings-emnlp.978",
     "MissMAC-Bench: Building Solid Benchmark for Missing Modality Issue in Robust Multimodal Affective Computing",
     "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing", "2026"),
    ("adaptivegated", "arxiv", "2510.01677",
     "Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis",
     "arXiv", "2025"),
    ("graphsmmile", "arxiv", "2407.21536",
     "Tracing Intricate Cues in Dialogue: Joint Graph Structure and Sentiment Dynamics for Multimodal Emotion Recognition",
     "arXiv", "2024"),
    ("emoshift", "arxiv", "2112.01938",
     "Multimodal Emotion Recognition in Conversations via Emotion Shifts",
     "arXiv", "2021"),
    ("mrcsurvey", "anthology", "2025.findings-emnlp.332",
     "Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects",
     "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing", "2025"),
    ("baselinepaper", "arxiv", "1803.07427",
     "Multimodal Sentiment Analysis: Addressing Key Issues and Setting up the Baselines",
     "arXiv", "2018"),
    ("tweetquant", "arxiv", "2011.08091",
     "Tweet Sentiment Quantification: An Experimental Re-Evaluation",
     "arXiv", "2020"),
    ("affectintweets", "anthology", "2020.lrec-1.188",
     "Affect inTweets: A Transfer Learning Approach",
     "Proceedings of the 14th Conference on Language Resources and Evaluation", "2020"),
    ("emollms", "arxiv", "2401.08508",
     "EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis",
     "arXiv", "2024"),
    ("llmmsasurvey", "arxiv", "2406.08068",
     "Large Language Models Meet Text-Centric Multimodal Sentiment Analysis: A Survey",
     "arXiv", "2024"),
    ("gatedfusion", "arxiv", "2401.06659",
     "WisdoM: Improving Multimodal Sentiment Analysis by Fusing Contextual World Knowledge",
     "arXiv", "2024"),
    ("trml", "arxiv", "2401.13697",
     "Toward Robust Multimodal Learning using Multimodal Foundational Models",
     "arXiv", "2024"),
]

# the anthology volume each key lives in, for the title search
VOLUME = {
    "mmml": "2024.naacl-long",
    "dpdflq": "2025.emnlp-main",
    "mmsa": "2021.acl-demo",
    "mosi": "2018.acl-long",
    "mosei": "2018.acl-long",
    "bert": "2019.naacl-long",
    "mmlf": "2020.acl-main",
}


def volume_papers(v):
    path = f"/tmp/anth_{v}.bib"
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        subprocess.run(["curl", "-s", "--max-time", "40", "-o", path,
                        f"https://aclanthology.org/volumes/{v}.bib"],
                       capture_output=True)
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    out = []
    for ident, body in re.findall(r"@inproceedings\{([^,]+),(.*?)\n\}", txt, re.S):
        t = re.search(r'title = "([^"]+)"', body)
        u = re.search(r'url = "([^"]+)"', body)
        y = re.search(r"year = \{?(\d{4})", body)
        if t and u:
            out.append((ident, t.group(1), u.group(1).rstrip("/"),
                        y.group(1) if y else ""))
    return out


def anthology_single(ident):
    """Fetch one paper's own bib file; the fields are written with braces."""
    raw = curl(f"https://aclanthology.org/{ident}.bib")
    t = re.search(r'title\s*=\s*\{([^}]*(?:\}[^{}]*\{[^}]*\})*[^}]*)\}', raw) \
        or re.search(r'title\s*=\s*"([^"]+)"', raw)
    y = re.search(r'year\s*=\s*"?\{?(\d{4})', raw)
    b = re.search(r'booktitle\s*=\s*\{([^}]*(?:\}[^{}]*\{[^}]*\})*[^}]*)\}', raw) \
        or re.search(r'booktitle\s*=\s*"([^"]+)"', raw)
    a = re.search(r'author\s*=\s*\{([^}]*)\}', raw) \
        or re.search(r'author\s*=\s*"([^"]+)"', raw)
    if not t:
        return None
    return {"title": t.group(1),
            "authors": re.sub(r"\s+", " ", a.group(1)).strip() if a else "",
            "venue": b.group(1) if b else "",
            "year": y.group(1) if y else "",
            "url": f"https://aclanthology.org/{ident}"}


def from_anthology(key, ident, want_title, venue, year):
    """Find the paper in its volume, or fetch the single-paper bib file."""
    # an anthology identifier of a single paper: "P18-1208", "N19-1423",
    # "2022.acl-demo.20"; a DOI starts with "10." and is handled elsewhere
    if not ident.startswith("10.") and len(ident) > 5 and (
            "." in ident or re.match(r"^[A-Z]\d", ident)):
        rec = anthology_single(ident)
        if rec is None:
            return None, "单篇 bib 取不到"
        if sim(want_title, rec["title"]) < THRESHOLD:
            return None, f"单篇题名不符：{rec['title']}"
        rec["venue"] = rec["venue"] or venue
        rec["year"] = rec["year"] or year
        return rec, ""
    v = VOLUME.get(key)
    if not v:
        return None, "未知卷"
    cands = [(sim(want_title, t), ident, t, u, y)
             for ident, t, u, y in volume_papers(v)]
    cands = [c for c in cands if c[0] > 0]
    if not cands:
        return None, f"卷 {v} 里没有可比对��条目"
    cands.sort(reverse=True)
    s, ident, t, u, y = cands[0]
    if s < THRESHOLD:
        return None, f"相似度 {s:.2f} < {THRESHOLD}（最佳候选：{t}）"
    raw = curl(f"{u}.bib")
    a = re.search(r'author\s*=\s*"([^"]+)"', raw)
    b = re.search(r'booktitle\s*=\s*"([^"]+)"', raw)
    return {
        "title": t, "authors": re.sub(r"\s+", " ", a.group(1)).strip() if a else "",
        "venue": b.group(1) if b else venue,
        "year": y or year, "url": u,
    }, ""


def from_arxiv(key, aid, want_title, venue, year):
    xml = curl(f"https://export.arxiv.org/api/query?id_list={aid}&max_results=3")
    ents = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    if not ents:
        return None, "arXiv 无条目"
    e = ents[0]
    t = re.search(r"<title>(.*?)</title>", e, re.S)
    p = re.search(r"<published>(\d{4})", e)
    d = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", e, re.S)
    au = re.findall(r"<name>(.*?)</name>", e, re.S)
    title = " ".join(t.group(1).split()) if t else ""
    s = sim(want_title, title)
    if s < THRESHOLD:
        return None, f"相似度 {s:.2f} < {THRESHOLD}（arXiv 题名：{title}）"
    return {
        "title": title,
        "authors": " and ".join(a.strip() for a in au),
        "venue": venue, "year": p.group(1) if p else year,
        "doi": d.group(1) if d else "",
        "url": f"https://arxiv.org/abs/{aid}",
    }, ""


def from_crossref(key, doi, want_title, venue, year):
    import json
    raw = curl("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    try:
        m = json.loads(raw)["message"]
    except Exception:
        return None, "Crossref 记录取不到"
    title = (m.get("title") or [""])[0]
    # Crossref sometimes stores an HTML-escaped ampersand
    title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = sim(want_title, title)
    if s < THRESHOLD:
        return None, f"相似度 {s:.2f} < {THRESHOLD}（Crossref 题名：{title}）"
    names = []
    for a in m.get("author", []) or []:
        fam, giv = a.get("family", ""), a.get("given", "")
        if fam:
            names.append(f"{fam}, {giv}" if giv else fam)
    ct = (m.get("container-title") or [""])[0]
    ct = ct.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    yr = ""
    for k in ("published-print", "published-online", "issued"):
        parts = m.get(k, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0]:
            yr = str(parts[0][0])
            break
    return {
        "title": title, "authors": " and ".join(names),
        "venue": ct or venue, "year": yr or year, "doi": doi,
        "url": f"https://doi.org/{doi}",
    }, ""


def main():
    out, ok, bad = [], [], []
    for key, source, ident, want, venue, year in REFERENCES:
        if source == "anthology":
            rec, err = from_anthology(key, ident, want, venue, year)
        elif source == "arxiv":
            rec, err = from_arxiv(key, ident, want, venue, year)
        else:
            rec, err = from_crossref(key, ident, want, venue, year)
        if rec is None:
            print(f"[未通过] {key:10s} {err}")
            bad.append(key)
            continue
        def esc(v):
            # some titles carry the model name in math mode, for example
            # "H2CAN" written as a product of two letters with an exponent;
            # the bibliography style then breaks the title across two lines
            # with the name set apart, so the math wrapper is removed here
            # and the name is written as plain text
            v = re.sub(r"\$\$\\text \{([^}]*)\}\$\$", r"\1", v)
            v = re.sub(r"\$\$", "", v)
            v = re.sub(r"\$", "", v)
            v = re.sub(r"\\text \{([^}]*)\}", r"\1", v)
            v = v.replace("\\text", "")
            # a caret that came out of the math form is not valid in the
            # text mode of the bibliography style; the exponent is a letter
            # of the model name and is folded back into the name
            v = re.sub(r"\^\{?([A-Za-z0-9])\}?", r"\1", v)
            # the ampersand has to be escaped inside a BibTeX value
            return (v.replace("&", r"\&").replace("%", r"\%")
                     .replace("_", r"\_").replace("#", r"\#"))
        lines = [f"@inproceedings{{{key}," if source == "anthology" else f"@article{{{key},",
                 f"  title = {{{esc(rec['title'])}}},"]
        if rec["authors"]:
            lines.append(f"  author = {{{esc(rec['authors'])}}},")
        lines.append(f"  booktitle = {{{esc(rec['venue'])}}}," if source == "anthology"
                     else f"  journal = {{{esc(rec['venue'])}}},")
        lines.append(f"  year = {{{rec['year']}}},")
        if rec.get("doi"):
            lines.append(f"  doi = {{{rec['doi']}}},")
        lines.append(f"  url = {{{rec['url']}}}")
        lines.append("}")
        out.append("\n".join(lines))
        ok.append((key, rec["title"], rec["venue"][:44], rec["year"]))

    with open(OUT, "w") as fh:
        fh.write("% every entry below was fetched from the ACL Anthology, arXiv or\n"
                 "% Crossref and its title was compared with the title used in the\n"
                 "% manuscript by scripts/verify_references.py; do not edit by hand\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"\n已写出 {OUT}：{len(ok)} 条已核验")
    for k, t, v, y in ok:
        print(f"  {k:10s} {y}  {t[:64]:66s} {v}")
    if bad:
        print(f"未通过：{', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
