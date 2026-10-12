"""Authenticated Modal inference, with scale-to-zero defaults."""

import modal
from dfip.config import settings
from dfip.cloud.images import image, julia_image, sglang_image, volume as research

app = modal.App(settings().app_name)


def initialize(model_id):
    import sys

    sys.path.insert(0, "/opt/dfip")
    from dfip.runtime.engine import DecisionEngine

    return DecisionEngine(model_id)


@app.cls(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": research},
    timeout=1200,
    scaledown_window=60,
    max_containers=2,
)
class Small:
    model_id: str = modal.parameter(default="LiquidAI/d1-3B")

    @modal.enter()
    def load(self):
        if self.model_id not in ("LiquidAI/d1-3B", "Cloudflare/clef-flash"):
            raise ValueError("Unsupported small checkpoint")
        self.engine = initialize(self.model_id)

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()


@app.cls(
    image=julia_image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    volumes={"/research": research},
    timeout=1200,
    scaledown_window=60,
    max_containers=2,
)
class Julia:
    @modal.enter()
    def load(self):
        self.engine = initialize("SupersonicLabs/Julia-1")

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()


@app.cls(
    image=image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": research},
    timeout=1200,
    scaledown_window=60,
    max_containers=1,
)
class Decider:
    @modal.enter()
    def load(self):
        self.engine = initialize("perplexity-ai/pplx-decider-v1.1-27b")

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()


@app.cls(
    image=sglang_image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": research},
    timeout=1200,
    scaledown_window=60,
    max_containers=1,
)
@modal.concurrent(max_inputs=64, target_inputs=64)
class Security:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/opt/dfip")
        from dfip.runtime.sglang_runtime import SGLangDecisionEngine

        self.engine = SGLangDecisionEngine(head_mode="native", concurrent=True)

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()

    @modal.exit()
    def stop(self):
        if hasattr(self, "engine"):
            self.engine.close()
