"""Record complete package inventories using CPU workers in the study images."""
import hashlib
import json
from pathlib import Path
import modal
from dfip.cloud.fast_baseline import fast_image
from dfip.cloud.images import sglang_image

app = modal.App('dfip-study-environment-inventory')


def inventory():
    import importlib.metadata
    import subprocess
    import sys
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
    result = {'python': sys.version, 'packages': packages,
              'scope': 'CPU-only inventory of the same image definitions, not an extra GPU benchmark round'}
    secondary = Path('/opt/sglang/bin/python')
    if secondary.exists():
        code = 'import importlib.metadata,json,sys; print(json.dumps({"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())}))'
        result['sglang_interpreter'] = json.loads(subprocess.check_output([str(secondary), '-c', code]))
    return result


@app.function(image=fast_image, cpu=1, max_containers=1, timeout=120)
def qwen_inventory():
    return inventory()


@app.function(image=sglang_image, cpu=1, max_containers=1, timeout=120)
def sglang_inventory():
    return inventory()


@app.local_entrypoint()
def main(commit: str = 'unknown'):
    result = {'source_commit': commit, 'inventory_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fast_image': qwen_inventory.remote(), 'sglang_image': sglang_inventory.remote()}
    dest = Path('benchmarks/results/strengthened/environment-inventory.json')
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'saved': str(dest), 'fast_packages': len(result['fast_image']['packages']),
                      'sglang_packages': len(result['sglang_image']['packages'])}))
