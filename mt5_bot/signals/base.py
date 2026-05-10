from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import ExternalSignal


class SignalSource(ABC):
    @abstractmethod
    def poll(self) -> List[ExternalSignal]:
        raise NotImplementedError

    def close(self) -> None:
        return None

