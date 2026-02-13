from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .data_types import Turn

@dataclass
class TurnBuffer:
    maxlen: int = 50
    turns: List[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.maxlen:
            self.turns = self.turns[-self.maxlen:]

    def last(self, n: int) -> List[Turn]:
        return self.turns[-n:]