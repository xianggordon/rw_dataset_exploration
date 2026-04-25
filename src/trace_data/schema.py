from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Turn:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class Trajectory:
    trajectory_id: str
    is_hacked: bool
    categories: list[str]
    turns: list[Turn]
    num_turns: int
    raw_conversation: str
    raw_label: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d
