"""Sequential GPU studies of released decision contracts and shared scheduling."""
import json
from pathlib import Path

import modal

from dfip.cloud.images import prepare_image, volume, PACKAGE
from dfip.cloud.fast_baseline import fast_base
from dfip.config import settings
from dfip.contract_models import CONTRACT_MODELS, contract_model

app = modal.App(settings().app_name + "-contracts")
image = fast_base.pip_install("gliner2[local]==2.1.0", "peft==0.21.2", "sentencepiece", "protobuf").add_local_dir(
    str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"]
)


@app.function(image=prepare_image, cpu=4, volumes={"/research": volume}, timeout=7200)
def prepare(alias):
    from dfip.cloud.contracts_prepare import download
    return download(alias)


def run(alias, action, commit, round_id):
    from dfip.runtime.contract_study import study

    volume.reload()
    result = study(alias, action, commit, round_id)
    folder = Path("/research/results/contracts"); folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{alias}-{action}-{round_id}-{commit[:12]}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    # torch.__version__ is a string subclass whose pickle imports torch on the
    # client. Transport only JSON primitives; the client needs no GPU packages.
    return json.loads(json.dumps(result))


@app.function(image=image, gpu="L40S", cpu=4, memory=49152, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def small(alias, action, commit, round_id):
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
            if spec.adapter == "jev":
                raise ValueError("Use dfip.cloud.contracts_jev for the vLLM backend")
            result = small.remote(spec.alias, action, commit, round_id)
        dest = folder / f"{spec.alias}-{action}-{round_id}-{commit[:12]}.json"
        dest.write_text(json.dumps(result, indent=2))
        print(json.dumps({"saved": str(dest), "status": result.get("status", "prepared")}), flush=True)
