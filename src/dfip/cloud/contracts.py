"""Sequential GPU studies of released decision contracts and shared scheduling."""
import json
from pathlib import Path

import modal

from dfip.cloud.images import prepare_image, volume, ENV, PACKAGE
from dfip.cloud.fast_baseline import fast_base
from dfip.config import settings
from dfip.contract_models import CONTRACT_MODELS, contract_model

app = modal.App(settings().app_name + "-contracts")
image = fast_base.pip_install("gliner2[local]==2.1.0", "peft==0.21.2", "sentencepiece", "protobuf").add_local_dir(
    str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"]
)
jev_image = modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12").entrypoint([]).pip_install(
    "vllm==0.31.0"
).env(ENV).add_local_dir(
    str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"]
)


@app.function(image=prepare_image, cpu=4, volumes={"/research": volume}, timeout=7200)
def prepare(alias):
    import time
    from huggingface_hub import snapshot_download

    spec = contract_model(alias)
    started = time.monotonic()
    path = snapshot_download(spec.model_id, revision=spec.revision,
                             allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.jinja", "LICENSE", "NOTICE"],
                             ignore_patterns=["reports/*", "assets/*", "videos/*", "vl/*"])
    result = {"model_id": spec.model_id, "revision": spec.revision, "path": path,
              "download_seconds": time.monotonic() - started}
    folder = Path("/research/prepared/contracts"); folder.mkdir(parents=True, exist_ok=True)
    (folder / (alias + ".json")).write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


def run(alias, action, commit, round_id):
    from dfip.runtime.contract_study import study

    volume.reload()
    result = study(alias, action, commit, round_id)
    folder = Path("/research/results/contracts"); folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{alias}-{action}-{round_id}-{commit[:12]}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=image, gpu="L40S", cpu=4, memory=49152, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def small(alias, action, commit, round_id):
    return run(alias, action, commit, round_id)


@app.function(image=jev_image, gpu="L40S", cpu=8, memory=98304, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def jev_small(alias, action, commit, round_id):
    return run(alias, action, commit, round_id)


@app.function(image=jev_image, gpu="B200", cpu=8, memory=131072, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def jev_big(alias, action, commit, round_id):
    return run(alias, action, commit, round_id)


@app.local_entrypoint()
def main(action: str = "prepare", model: str = "all", commit: str = "unknown", round_id: int = 0,
         output: str = "benchmarks/results/contracts"):
    selected = CONTRACT_MODELS if model == "all" else (contract_model(model),)
    folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
    for spec in selected:
        if action == "prepare":
            result = prepare.remote(spec.alias)
        else:
            fn = jev_big if spec.gpu == "B200" else jev_small if spec.adapter == "jev" else small
            result = fn.remote(spec.alias, action, commit, round_id)
        dest = folder / f"{spec.alias}-{action}-{round_id}-{commit[:12]}.json"
        dest.write_text(json.dumps(result, indent=2))
        print(json.dumps({"saved": str(dest), "status": result.get("status", "prepared")}), flush=True)
