"""Load examples/environment.json (or path arg) and print reasoning_output JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from peter_pan.agent_brain.reasoning_factory import create_reasoning_agent
from peter_pan.config import load_config
from peter_pan.perception.environment_builder import load_environment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("env_json", nargs="?", default=str(ROOT / "examples" / "environment.json"))
    args = ap.parse_args()
    env = load_environment(args.env_json)
    agent = create_reasoning_agent(load_config())
    out = agent.decision(env)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
