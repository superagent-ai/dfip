"""Pinned checkpoint registry. Adding a tag alone does not imply compatibility."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    alias: str
    model_id: str
    revision: str
    modal_class: str
    gpu: str


MODELS = (
    Model(
        "julia",
        "SupersonicLabs/Julia-1",
        "a85b127321d580d65176c89ced8273f305745d85",
        "Julia",
        "L40S",
    ),
    Model(
        "liquid",
        "LiquidAI/d1-3B",
        "051bcc464b01b9f92942b364d9586b0ef5912432",
        "Small",
        "L40S",
    ),
    Model(
        "clef",
        "Cloudflare/clef-flash",
        "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "Small",
        "L40S",
    ),
    Model(
        "decider",
        "perplexity-ai/pplx-decider-v1.1-27b",
        "6195aa55a72ae21e503c8a60198f7b390d42adb5",
        "Decider",
        "B200",
    ),
    Model(
        "security",
        "superagent-ai/security-one-27b",
        "1ae09b6bc17a87332341bcd95ea42fa6b5e6b399",
        "Security",
        "B200",
    ),
)


def get_model(name: str) -> Model:
    for model in MODELS:
        if name in (model.alias, model.model_id):
            return model
    raise ValueError(
        f"Unknown model {name!r}. Choose: {', '.join(m.alias for m in MODELS)}"
    )
