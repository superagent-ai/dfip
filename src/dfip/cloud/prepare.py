"""Download pinned checkpoints directly into a Modal volume."""

import modal
from dfip.config import settings
from dfip.models import MODELS, get_model
from dfip.cloud.images import prepare_image, volume

app = modal.App(settings().app_name + "-prepare")
secrets = [modal.Secret.from_name(settings().hf_secret)] if settings().hf_secret else []


@app.function(
    image=prepare_image,
    volumes={"/research": volume},
    secrets=secrets,
    timeout=3600,
    cpu=4,
)
def prepare(model_name: str):
    import json
    import os
    import time
    from pathlib import Path
    from huggingface_hub import snapshot_download

    model = get_model(model_name)
    started = time.monotonic()
    path = snapshot_download(
        model.model_id, revision=model.revision, token=os.environ.get("HF_TOKEN")
    )
    result = {
        "model_id": model.model_id,
        "revision": model.revision,
        "path": path,
        "download_seconds": time.monotonic() - started,
    }
    directory = Path("/research/prepared")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (model.model_id.replace("/", "__") + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(model: str = "liquid"):
    import json

    selected = MODELS if model == "all" else [get_model(model)]
    for spec in selected:
        print(json.dumps(prepare.remote(spec.alias), indent=2))
