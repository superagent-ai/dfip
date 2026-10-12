"""Opt-in experimental app using the same images and policy as the v4 pilot."""
import modal
from dfip.cloud.fast_baseline import fast_image
from dfip.cloud.images import volume
from dfip.config import settings

app = modal.App(settings().app_name)


def initialize(model_id):
    from dfip.runtime.adaptive_engine import AdaptiveDecisionEngine
    volume.reload()
    return AdaptiveDecisionEngine(model_id)


@app.cls(image=fast_image, gpu='L40S', cpu=4, memory=32768,
         volumes={'/research': volume}, max_containers=1, scaledown_window=60, timeout=1200)
@modal.concurrent(max_inputs=64, target_inputs=64)
class Small:
    model_id: str = modal.parameter(default='Cloudflare/clef-flash')

    @modal.enter()
    def load(self):
        if self.model_id != 'Cloudflare/clef-flash':
            raise ValueError('This experimental Small class supports Clef only')
        self.engine = initialize(self.model_id)

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()


@app.cls(image=fast_image, gpu='B200', cpu=8, memory=98304,
         volumes={'/research': volume}, max_containers=1, scaledown_window=60, timeout=1200)
@modal.concurrent(max_inputs=64, target_inputs=64)
class Decider:
    @modal.enter()
    def load(self):
        self.engine = initialize('perplexity-ai/pplx-decider-v1.1-27b')

    @modal.method()
    def predict(self, records):
        return self.engine.predict(records)

    @modal.method()
    def diagnostics(self):
        return self.engine.diagnostics()
