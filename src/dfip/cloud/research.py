"""Run review-paper experiments on Modal, keeping checkpoints and corpora remote."""

import json
from pathlib import Path
import modal
from dfip.cloud.images import image, julia_image, volume
from dfip.config import settings
from dfip.models import get_model

app = modal.App(settings().app_name + "-paper")


def run(model, round_id, repetitions, fidelity, commit):
    from dfip.runtime.research import round_run

    volume.reload()
    result = round_run(model, round_id, repetitions, fidelity, commit)
    folder = Path("/research/results/paper-v1")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{model}-{round_id}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=julia_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
)
def julia(model, round_id, repetitions, fidelity, commit):
    return run(model, round_id, repetitions, fidelity, commit)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
)
def small(model, round_id, repetitions, fidelity, commit):
    return run(model, round_id, repetitions, fidelity, commit)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
)
def big(model, round_id, repetitions, fidelity, commit):
    return run(model, round_id, repetitions, fidelity, commit)


@app.local_entrypoint()
def main(
    model: str = "liquid",
    rounds: int = 5,
    repetitions: int = 50,
    output: str = "benchmarks/results",
    commit: str = "unknown",
):
    spec = get_model(model)
    if spec.alias == "security":
        raise ValueError("Use the Security-One research entry point")
    function = (
        julia if spec.alias == "julia" else big if spec.alias == "decider" else small
    )
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    args = [(spec.alias, i, repetitions, i == 0, commit) for i in range(rounds)]
    for result in function.starmap(args, order_outputs=False):
        dest = folder / f"{spec.alias}-round-{result['round']}.json"
        dest.write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {"saved": str(dest), "model": spec.alias, "round": result["round"]}
            ),
            flush=True,
        )


@app.function(image=image, volumes={"/research": volume}, timeout=600)
def probe_data():
    from dfip.runtime.research import banking_records, new_records
    import transformers.models.qwen3_5.modeling_qwen3_5 as q

    rows = banking_records()
    extra = new_records()
    return {
        "banking_records": len(rows),
        "expanded_records": len(extra),
        "sample_id": rows[0]["id"],
        "native_causal_conv_available": q.causal_conv1d_fn is not None,
    }
