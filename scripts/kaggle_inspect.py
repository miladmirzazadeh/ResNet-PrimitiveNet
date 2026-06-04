"""
=============================================================================
 KAGGLE INSPECTION — verify the ArchCAD JSON structure (run THIS, paste output)
=============================================================================

WHY: the parser in vtrue/archcad.py is written against the documented schema.
This script confirms the ACTUAL field names, entity types, and `semantic` string
values so we lock the mapping before training. Kaggle's network downloads from
HF much faster than a laptop.

HOW (on kaggle.com, a NEW notebook, Internet = ON):
  1. Add your HF token as a Kaggle Secret named  HF_TOKEN
     (Notebook -> Add-ons -> Secrets -> Add: label HF_TOKEN, value = your hf_... token).
     Do NOT paste the token into the code or share it with anyone.
  2. Paste this whole file into one cell and Run.
  3. Copy the printed output back to me. That's all I need.

It downloads ONLY json.zip (~393 MB) and inspects a sample — nothing is uploaded.
"""

# ---- cell ----
import os, io, zipfile, json, collections, random

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

from huggingface_hub import hf_hub_download

REPO = "jackluoluo/ArchCAD"
path = None
for cand in ("data/json.zip", "json.zip"):
    try:
        path = hf_hub_download(repo_id=REPO, filename=cand, repo_type="dataset",
                               token=os.environ["HF_TOKEN"])
        print("got:", cand, "->", path)
        break
    except Exception as e:
        print("not at", cand, "->", repr(e)[:120])
assert path, "could not download json.zip — check token / gated access"

# ---- cell ----
zf = zipfile.ZipFile(path)
names = [n for n in zf.namelist() if n.lower().endswith(".json")]
print("TOTAL json files in archive:", len(names))
print("archive top-level entries (sample):", sorted({n.split('/')[0] for n in zf.namelist()})[:10])

sample = random.Random(0).sample(names, min(40, len(names)))
types = collections.Counter()
sem_strings = collections.Counter()
sem_is_int = 0; sem_is_str = 0
key_sets = collections.Counter()
prim_counts = []
coord_min = 1e18; coord_max = -1e18
example_printed = 0

for nm in sample:
    data = json.loads(zf.read(nm).decode("utf-8", "ignore"))
    if isinstance(data, dict):
        data = data.get("primitives") or data.get("entities") or data.get("elements") or []
    prim_counts.append(len(data))
    for e in data:
        if not isinstance(e, dict):
            continue
        key_sets[tuple(sorted(e.keys()))] += 1
        types[str(e.get("type"))] += 1
        s = e.get("semantic", e.get("semantic_id"))
        if isinstance(s, str):
            sem_is_str += 1; sem_strings[s] += 1
        elif isinstance(s, (int, float)):
            sem_is_int += 1; sem_strings[int(s)] += 1
        for ck in ("start", "end", "center"):
            v = e.get(ck)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                coord_min = min(coord_min, float(v[0]), float(v[1]))
                coord_max = max(coord_max, float(v[0]), float(v[1]))
    if example_printed < 2 and data:
        print(f"\n--- EXAMPLE PRIMITIVES from {nm} ---")
        for e in data[:4]:
            print(json.dumps(e, ensure_ascii=False))
        example_printed += 1

print("\n==== STRUCTURE REPORT (paste all of this back) ====")
print("primitives/file: min %d  max %d  mean %.0f" %
      (min(prim_counts), max(prim_counts), sum(prim_counts) / len(prim_counts)))
print("coord range: %.1f .. %.1f" % (coord_min, coord_max))
print("\nENTITY 'type' histogram:", dict(types))
print("\nsemantic stored as: str=%d  int=%d" % (sem_is_str, sem_is_int))
print("\nSEMANTIC value histogram (the labels — names or ids):")
for k, v in sem_strings.most_common(40):
    print("  %-22s %d" % (str(k), v))
print("\nDISTINCT key-sets per primitive (field names present):")
for ks, c in key_sets.most_common(8):
    print("  %3d x  %s" % (c, list(ks)))
print("==== END REPORT ====")
