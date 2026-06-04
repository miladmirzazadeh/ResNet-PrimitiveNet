#!/usr/bin/env bash
# Hands-off pipeline: data -> train -> eval -> upload results to HF -> STOP the pod.
# Run from inside the cloned repo:
#   export HF_TOKEN=hf_...
#   export RUNPOD_API_KEY=...                 # runpod.io > Settings > API Keys (read+write)
#   nohup bash run_all.sh > /workspace/pipeline.log 2>&1 &
#   sleep 10; tail -f /workspace/pipeline.log
#
# Results land in the private HF repo $HF_RESULT_REPO; the pod stops itself when done
# (or on any failure, via the EXIT trap) so you don't have to wake up to turn it off.

export RUN=${RUN:-/workspace/runs/v1}
export HF_RESULT_REPO=${HF_RESULT_REPO:-miladmirza/vtrue-results}
export DATA=${DATA:-/root/archcad}            # local SSD: fast extract, split, and per-epoch reads
TB=${TIME_BUDGET:-5.5}                         # training-loop cap in hours (override: export TIME_BUDGET=...)
EP=${EPOCHS:-45}
mkdir -p "$RUN"
cd "$(cd "$(dirname "$0")" && pwd)"           # repo root

finish() {
  echo "=== FINISH: upload results + stop pod ==="
  cp /workspace/pipeline.log "$RUN/" 2>/dev/null
  python - <<'PY'
import os, json, urllib.request
from huggingface_hub import HfApi
RUN=os.environ["RUN"]; REPO=os.environ["HF_RESULT_REPO"]; tok=os.environ.get("HF_TOKEN")
try:
    api=HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True, token=tok)
    api.upload_folder(folder_path=RUN, repo_id=REPO, repo_type="model", token=tok,
                      ignore_patterns=["last.pt"])
    print("uploaded ->", REPO)
except Exception as e:
    print("upload failed:", repr(e))
key=os.environ.get("RUNPOD_API_KEY",""); pod=os.environ.get("RUNPOD_POD_ID","")
try:
    body=json.dumps({"query":"mutation($id:String!){podStop(input:{podId:$id}){id desiredStatus}}",
                     "variables":{"id":pod}})
    req=urllib.request.Request("https://api.runpod.io/graphql?api_key="+key,
                               data=body.encode(), headers={"content-type":"application/json"})
    print("pod stop ->", urllib.request.urlopen(req, timeout=30).read().decode())
except Exception as e:
    print("pod stop failed:", repr(e))
PY
}
trap finish EXIT
set -x

bash setup_runpod.sh
pip install -q timm

# ---- data (skip if already split) ----
if [ -z "$(ls -A "$DATA/train" 2>/dev/null)" ]; then
  python scripts/download_archcad.py --out /workspace/data/raw
  ZIP=$(find /workspace/data/raw -name json.zip | head -1)
  mkdir -p /root/ex
  unzip -q -o "$ZIP" -d /root/ex
  python scripts/split_archcad.py --src /root/ex --out "$DATA"
fi

# ---- train ----
python -m vtrue.train --data "$DATA" --out "$RUN" \
  --preset medium --batch 32 --max-prims 1024 --epochs "$EP" --workers 32 \
  --auto-weight --flaw-frac 0.2 --lr 4e-4 --time-budget "$TB"

# ---- eval ----
python -m vtrue.eval --weights "$RUN/best.pt" --data "$DATA" \
  --split val --json "$RUN/metrics.json"
