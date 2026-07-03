"""
export_transcripts.py - Export final JSONs to the frontend ingest contract.

Reads every setting from pipeline_config.json (repo root) — layout, id
template, timestamp format, field names — so the ingest side can be matched by
editing config, not code. Idempotent/upsert-friendly: stable ids, output only
rewritten when the source is newer, manifest.jsonl gets one line per file
written this run (delta ingest: consume lines after your last offset).

Per exported file:
{
  "id": "downloads/vaisesika-dasa/<basename>.mp3",   # stable, from id_template
  "speaker": "vaisesika-dasa",
  "title": "<cleaned from filename>",
  "category": "lecture",
  "language": "en",
  "duration": 62,                                     # integer minutes
  "sections": [{"speaker","start","end","text"}],     # MM:SS timestamps
  "pipeline": { provenance the app can ignore }
}

Usage:
    python 06_postprocessing/export_transcripts.py [--limit N] [--force]
"""
import os, sys, json, re, glob, hashlib, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(REPO, "pipeline_config.json")))
EXP = CFG["export"]
F = EXP["field_names"]


def hms_to_sec(ts):
    if isinstance(ts, (int, float)):        # recovered files store float seconds
        return int(round(ts))
    p = [int(float(x)) for x in ts.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def fmt_ts(sec):
    if EXP["timestamp_format"] == "MM:SS":
        return f"{sec // 60:02d}:{sec % 60:02d}"
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def clean_title(stem):
    t = re.sub(r"-\d{8,}$", "", stem)                      # trailing fb/media id
    t = re.sub(r"^(fts_fb_)?\d{4}([-_]\d{2})?(_part-\d+)?_?", "", t)  # date/part prefix
    t = t.replace("_", " ")
    t = re.sub(r"\s*(by|By)\s+H\.?[GH]\.?\s.*$", "", t)    # "by HG ..." suffix
    t = re.sub(r"\s*\|\|.*$", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -_·")
    return t or stem


def main():
    limit = 0
    force = "--force" in sys.argv
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    slug = CFG["corpus"]["main_speaker_slug"]
    main_name = CFG["corpus"]["main_speaker_name"]
    src_dir = os.path.join(REPO, CFG["pipeline_outputs"]["final_json_dir"])
    fuzz_dir = os.path.join(REPO, CFG["pipeline_outputs"]["fuzz_merged_dir"])
    out_root = os.path.join(REPO, EXP["out_root"])
    manifest = os.path.join(REPO, EXP["manifest"])
    os.makedirs(os.path.dirname(manifest), exist_ok=True)

    valid = set(CFG["speaker_slugs"]["valid"])
    if slug not in valid:
        sys.exit(f"[export] main speaker slug '{slug}' not in speaker_slugs.valid")

    written = skipped = 0
    with open(manifest, "a") as mf:
        for src in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
            stem = os.path.basename(src)[:-5]
            dst = EXP["layout"].format(out_root=out_root, speaker_slug=slug, basename=stem)
            if not force and os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                skipped += 1
                continue
            segs = json.load(open(src))
            sections, last_end = [], 0
            for s in segs:
                name = next(iter(s))
                body = s[name]
                a, b = hms_to_sec(body["start"]), hms_to_sec(body["end"])
                last_end = max(last_end, b)
                sec_speaker = slug if name == main_name else name
                if EXP.get("audience_label_slug") and name != main_name:
                    sec_speaker = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
                sections.append({F["section_speaker"]: sec_speaker,
                                 F["section_start"]: fmt_ts(a),
                                 F["section_end"]: fmt_ts(b),
                                 F["section_text"]: body["text"]})
            doc = {
                F["id"]: EXP["id_template"].format(speaker_slug=slug, basename=stem),
                F["speaker"]: slug,
                F["title"]: clean_title(stem),
                F["category"]: EXP["category_default"],
                F["language"]: EXP["language_default"],
                F["duration"]: round(last_end / 60),
                F["sections"]: sections,
                F["pipeline_meta"]: {
                    "stem": stem,
                    "whisper_only": os.path.exists(os.path.join(fuzz_dir, stem, "WHISPER_ONLY")),
                    "n_sections": len(sections),
                    "exported_at": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"),
                },
            }
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            blob = json.dumps(doc, indent=1, ensure_ascii=False)
            with open(dst, "w") as f:
                f.write(blob)
            mf.write(json.dumps({
                "id": doc[F["id"]], "path": os.path.relpath(dst, REPO),
                "sha1": hashlib.sha1(blob.encode()).hexdigest(),
                "updated_at": doc[F["pipeline_meta"]]["exported_at"],
            }) + "\n")
            written += 1
            if limit and written >= limit:
                break
    print(f"[export] wrote {written}, up-to-date {skipped} -> {out_root} (manifest: {manifest})")


if __name__ == "__main__":
    main()
