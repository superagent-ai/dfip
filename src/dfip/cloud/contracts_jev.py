"""Native vLLM research workers, built separately from tensor model workers."""
import json
from pathlib import Path

import modal

from dfip.cloud.images import volume, ENV, PACKAGE
from dfip.cloud.contracts import run
from dfip.config import settings
from dfip.contract_models import contract_model

app = modal.App(settings().app_name + "-contracts-jev")
image = modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12").entrypoint([]).pip_install(
    "vllm==0.31.0"
).env(ENV).add_local_dir(str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"])


@app.function(image=image, gpu="L40S", cpu=8, memory=98304, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def small(alias, action, commit, round_id):
    return run(alias, action, commit, round_id)


@app.function(image=image, gpu="B200", cpu=8, memory=131072, volumes={"/research": volume},
              timeout=3600, max_containers=1, single_use_containers=True)
def big(alias, action, commit, round_id):
    return run(alias, action, commit, round_id)


@app.local_entrypoint()
def main(action: str = "native", model: str = "jev9", commit: str = "unknown", round_id: int = 0):
    spec = contract_model(model)
    if spec.adapter != "jev":
        raise ValueError("This worker covers native vLLM decision LoRA models")
    result = (big if spec.gpu == "B200" else small).remote(spec.alias, action, commit, round_id)
    folder = Path("benchmarks/results/contracts"); folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{spec.alias}-{action}-{round_id}-{commit[:12]}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps({"saved": str(dest), "status": result["status"]}), flush=True)
