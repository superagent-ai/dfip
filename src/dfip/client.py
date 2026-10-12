"""Lightweight authenticated client. No weights or GPU libraries are loaded locally."""

from dfip.config import settings
from dfip.models import get_model
from dfip.schema import validate_records


class Client:
    def __init__(self, model="liquid", *, app_name=None, environment_name=None):
        self.model = get_model(model)
        self.app_name = app_name or settings().app_name
        self.environment_name = environment_name
        self._engine = None

    def _remote(self):
        if self._engine is None:
            import modal

            kwargs = (
                {"environment_name": self.environment_name}
                if self.environment_name
                else {}
            )
            cls = modal.Cls.from_name(self.app_name, self.model.modal_class, **kwargs)
            self._engine = (
                cls(model_id=self.model.model_id)
                if self.model.modal_class == "Small"
                else cls()
            )
        return self._engine

    def predict(self, records):
        validate_records(records)
        return self._remote().predict.remote(records)

    async def predict_async(self, records):
        validate_records(records)
        return await self._remote().predict.remote.aio(records)

    def diagnostics(self):
        return self._remote().diagnostics.remote()
