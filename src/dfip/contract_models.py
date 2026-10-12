"""Research-only checkpoints; inclusion does not imply client compatibility."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContractModel:
    alias: str
    model_id: str
    revision: str
    adapter: str
    gpu: str


CONTRACT_MODELS = (
    ContractModel("omni", "LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674", "omni", "L40S"),
    ContractModel("gliner", "fastino/GLiNER2.5-Decide", "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90", "gliner", "L40S"),
    ContractModel("d3", "vllm-sr/d3-flash", "f6075bff3c0f7d5704e353d0fb2ebf94888d1691", "d3", "L40S"),
    ContractModel("jev9", "autotrust/JEV-9B", "7ac669b8fb476ba926491e3d1c39f9330d68dbed", "jev", "L40S"),
    ContractModel("jev27", "autotrust/JEV-27B", "14b425269ed24452e0a5ad5a8f3e51d139647a88", "jev", "B200"),
)


def contract_model(alias):
    return next(m for m in CONTRACT_MODELS if alias in (m.alias, m.model_id))
