import json
from pathlib import Path
import modal
from dfip.cloud.images import sglang_image, volume
from dfip.config import settings

app = modal.App(settings().app_name + "-paper-security")


@app.function(
    image=sglang_image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": volume},
    timeout=7200,
    max_containers=1,
    single_use_containers=True,
)
def experiment(round_id, repetitions, fidelity, commit):
    from dfip.runtime.security_research import run

    volume.reload()
    result = run(round_id, repetitions, fidelity, commit)
    folder = Path("/research/results/paper-v2")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"security-{round_id}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    rounds: int = 5,
    repetitions: int = 50,
    output: str = "benchmarks/results/follow-up",
    commit: str = "unknown",
):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    for result in experiment.starmap(
        [(i, repetitions, i == 0, commit) for i in range(rounds)], order_outputs=False
    ):
        dest = folder / f"security-round-{result['round']}.json"
        dest.write_text(json.dumps(result, indent=2))
        print(json.dumps({"saved": str(dest)}), flush=True)
