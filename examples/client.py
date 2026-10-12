"""Run after dfip setup --model liquid."""

import json
from pathlib import Path
from dfip import Client

records = json.loads(Path(__file__).with_name("request.json").read_text())
client = Client("liquid")
print(json.dumps(client.predict(records), indent=2))
