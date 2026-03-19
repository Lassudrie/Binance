from __future__ import annotations

from pathlib import Path

from ofbot.config import AppConfig
from ofbot.replay.player import iter_replay_events
from ofbot.engine import PaperEngine


def run_replay(config: AppConfig, input_path: Path) -> Path:
    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        events = iter_replay_events(input_path)
        for event in events:
            engine.process_event(event)
        return engine.finalize()
    finally:
        engine.close()
