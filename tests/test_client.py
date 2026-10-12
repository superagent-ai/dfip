import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from dfip import Client
from dfip.cli import main
from dfip.models import MODELS, get_model
from dfip.schema import validate_records

REQUEST = json.loads((Path(__file__).parents[1] / "examples/request.json").read_text())


def test_import_and_cli_do_not_load_gpu_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dfip,sys; assert not any(x in sys.modules for x in ('torch','triton','transformers')); from dfip.cli import main; main(['models'])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 5


@pytest.mark.parametrize("model", MODELS)
def test_client_routes_each_model_to_correct_deployed_class(model):
    engine = MagicMock()
    engine.predict.remote.return_value = [{"answers": {"team": {"choice": "billing"}}}]
    remote_class = MagicMock(return_value=engine)
    with patch("modal.Cls.from_name", return_value=remote_class) as lookup:
        client = Client(model.alias, app_name="my-dfip", environment_name="staging")
        assert client.predict(REQUEST)[0]["answers"]["team"]["choice"] == "billing"
        lookup.assert_called_once_with(
            "my-dfip", model.modal_class, environment_name="staging"
        )
        remote_class.assert_called_once_with(
            **({"model_id": model.model_id} if model.modal_class == "Small" else {})
        )
        engine.predict.remote.assert_called_once_with(REQUEST)
        client.diagnostics()
        lookup.assert_called_once()


def test_async_client():
    async def predict(records):
        return records

    client = Client("liquid")
    client._engine = SimpleNamespace(
        predict=SimpleNamespace(remote=SimpleNamespace(aio=predict))
    )
    assert asyncio.run(client.predict_async(REQUEST)) == REQUEST


@pytest.mark.parametrize(
    "records",
    [
        [],
        {},
        [None],
        [{"state": "x", "questions": {}}],
        [{"state": "x", "questions": {"q": {"type": "choice", "criteria": []}}}],
        [
            {
                "state": "x",
                "questions": {"q": {"type": "score", "criteria": {"0": "low"}}},
            }
        ],
        [
            {
                "state": "x",
                "questions": {"q": {"type": "noul", "criteria": {"true": "yes"}}},
            }
        ],
        [{"state": float("nan"), "questions": {"q": {"type": "noul"}}}],
        [dict(REQUEST[0], images=["x"])],
    ],
)
def test_invalid_requests_fail_before_network(records):
    with patch("modal.Cls.from_name") as lookup:
        with pytest.raises(ValueError):
            Client().predict(records)
        lookup.assert_not_called()


def test_all_question_types_and_structured_state():
    record = {
        "state": {"ticket": "refund"},
        "questions": {
            "refund": {"type": "noul"},
            "urgency": {"type": "score", "criteria": ["low", "high"]},
            "team": REQUEST[0]["questions"]["team"],
        },
    }
    assert validate_records([record]) == [record]


def test_setup_does_not_deploy_after_failed_download():
    with patch("dfip.cli.modal_command", return_value=7) as invoke:
        assert main(["setup", "--model", "liquid"]) == 7
        invoke.assert_called_once_with(
            "run", "dfip.cloud.prepare", ["--model", "liquid"]
        )


def test_setup_deploys_after_successful_download():
    with patch("dfip.cli.modal_command", return_value=0) as invoke:
        assert main(["setup", "--model", "julia"]) == 0
        assert invoke.call_args_list[-1].args == ("deploy", "dfip.cloud.serve")


def test_unknown_model_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("untested-model")


@pytest.mark.parametrize("alias", ["julia", "liquid", "clef", "decider"])
def test_native_deployment_bypasses_acceleration_and_preserves_adapter_output(alias):
    import importlib.util

    expected = [{"answers": {"team": {"choice": "billing"}}}]
    adapter = SimpleNamespace(predict=MagicMock(return_value=expected), model=SimpleNamespace(lm_head=None))
    optimize = MagicMock(side_effect=AssertionError("Rejected acceleration must not run"))
    fake_dependencies = {
        "dfip.runtime.adapters": SimpleNamespace(load_adapter=MagicMock()),
        "dfip.runtime.optimization": SimpleNamespace(optimize=optimize),
    }
    spec = importlib.util.spec_from_file_location(
        "native_engine_test", Path(__file__).parents[1] / "src/dfip/runtime/engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_dependencies):
        spec.loader.exec_module(module)
    engine = module.DecisionEngine(get_model(alias).model_id, adapter=adapter)
    records = REQUEST if isinstance(REQUEST, list) else [REQUEST]
    assert engine.predict(records) == expected
    adapter.predict.assert_called_once_with(records)
    optimize.assert_not_called()
    assert engine.diagnostics()["optimization"]["mode"] == "native"
    assert engine.diagnostics()["graphs"] == []
