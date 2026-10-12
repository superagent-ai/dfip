"""CPU-only preparation, independent of GPU backend image builds."""
import json
from pathlib import Path
import time

import modal

from dfip.cloud.images import prepare_image, volume
from dfip.contract_models import CONTRACT_MODELS, contract_model

app = modal.App("dfip-study5-checkpoint-prepare")


def download(alias):
    from huggingface_hub import snapshot_download
    spec = contract_model(alias)
    started = time.monotonic()
    path = snapshot_download(spec.model_id, revision=spec.revision,
                             allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.jinja", "LICENSE", "NOTICE"],
                             ignore_patterns=["reports/*", "assets/*", "videos/*", "vl/*"])
    result = {"model_id": spec.model_id, "revision": spec.revision, "path": path,
              "download_seconds": time.monotonic()-started}
    folder = Path("/research/prepared/contracts"); folder.mkdir(parents=True, exist_ok=True)
    (folder / (alias + ".json")).write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=prepare_image, cpu=4, volumes={"/research": volume}, timeout=7200)
def prepare(alias):
    return download(alias)


@app.local_entrypoint()
def main(model: str = "all"):
    folder = Path("benchmarks/results/contracts"); folder.mkdir(parents=True, exist_ok=True)
    for spec in CONTRACT_MODELS if model == "all" else (contract_model(model),):
        result = prepare.remote(spec.alias)
        (folder / (spec.alias + "-prepared.json")).write_text(json.dumps(result, indent=2))
        print(json.dumps({"prepared": spec.alias, "seconds": result["download_seconds"]}), flush=True)
