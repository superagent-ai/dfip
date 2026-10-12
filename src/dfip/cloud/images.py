"""Pinned remote images and the user-owned checkpoint volume."""

from pathlib import Path
import modal
from dfip.config import settings

PACKAGE = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name(settings().volume_name, create_if_missing=True)
ENV = {
    "HF_HOME": "/research/hf",
    "TORCHINDUCTOR_CACHE_DIR": "/research/inductor",
    "TRITON_CACHE_DIR": "/research/triton",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONPATH": "/opt/dfip",
}
base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "huggingface_hub==1.6.0",
        "accelerate",
        "safetensors",
        "numpy",
        "pillow",
        "pyarrow",
        "requests",
        "scipy",
    )
    .pip_install("fla-core==0.5.2")
    .env(ENV)
)
image = base.add_local_dir(
    str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"]
)
julia_image = base.pip_install("transformers==5.0.0").add_local_dir(
    str(PACKAGE), remote_path="/opt/dfip/dfip", ignore=["**/__pycache__/**", "**/*.pyc"]
)
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.19-cu130")
    .entrypoint([])
    .env(dict(ENV, TRITON_CACHE_DIR="/research/sglang-triton"))
    .add_local_dir(
        str(PACKAGE),
        remote_path="/opt/dfip/dfip",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)
prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.6.0")
    .env(ENV)
    .add_local_dir(
        str(PACKAGE),
        remote_path="/opt/dfip/dfip",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)
