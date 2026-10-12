"""Account-independent configuration, read at invocation time."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str
    volume_name: str
    hf_secret: str | None


def settings() -> Settings:
    return Settings(
        app_name=os.environ.get("DFIP_APP", "dfip-inference"),
        volume_name=os.environ.get("DFIP_VOLUME", "dfip-models-v1"),
        hf_secret=os.environ.get("DFIP_HF_SECRET") or None,
    )
