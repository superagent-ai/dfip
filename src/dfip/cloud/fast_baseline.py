"""Optional upstream CUDA convolution baseline, built remotely from pinned source."""

import modal
from dfip.cloud.images import ENV, PACKAGE, volume
from dfip.cloud.research import run
from dfip.config import settings

app = modal.App(settings().app_name + "-paper-fast-conv")
fast_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "clang")
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
        "packaging",
        "wheel",
        "ninja",
    )
    .pip_install("fla-core==0.5.2")
    .env(
        dict(
            ENV,
            CAUSAL_CONV1D_FORCE_BUILD="TRUE",
            MAX_JOBS="4",
            TORCH_CUDA_ARCH_LIST="8.9;10.0",
            CUDA_HOME="/usr/local/cuda",
            CC="gcc",
            CXX="g++",
        )
    )
    .pip_install("causal-conv1d==1.7.0", extra_options="--no-build-isolation")
    .add_local_dir(
        str(PACKAGE),
        remote_path="/opt/dfip/dfip",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


@app.function(
    image=fast_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=1,
    single_use_containers=True,
)
def small(model, round_id, repetitions, fidelity, commit):
    return run(model, round_id, repetitions, fidelity, commit, variant="fast")


@app.function(
    image=fast_image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=1,
    single_use_containers=True,
)
def big(model, round_id, repetitions, fidelity, commit):
    return run(model, round_id, repetitions, fidelity, commit, variant="fast")


@app.local_entrypoint()
def main(
    model: str = "clef",
    rounds: int = 5,
    repetitions: int = 50,
    output: str = "benchmarks/results/fast-conv",
    commit: str = "unknown",
    action: str = "benchmark",
):
    import json
    from pathlib import Path

    if action == "probe":
        print(json.dumps(probe.remote()), flush=True)
        return
    if model not in ("clef", "decider"):
        raise ValueError("Fast baseline currently covers Clef and Decider")
    fn = small if model == "clef" else big
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    for result in fn.starmap(
        [(model, i, repetitions, i == 0, commit) for i in range(rounds)],
        order_outputs=False,
    ):
        dest = folder / f"{model}-fast-round-{result['round']}.json"
        dest.write_text(json.dumps(result, indent=2))
        print(json.dumps({"saved": str(dest)}), flush=True)


@app.function(image=fast_image, cpu=4, timeout=600, max_containers=1)
def probe():
    import causal_conv1d
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as q
    return {"torch": torch.__version__, "causal_conv1d": causal_conv1d.__version__,
            "native_causal_conv_available": q.causal_conv1d_fn is not None}
