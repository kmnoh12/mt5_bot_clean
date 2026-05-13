from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderPermissionState:
    allow_new_entries: bool
    allow_protection_modify: bool
    allow_normal_close: bool
    allow_emergency_close: bool

    @classmethod
    def all_allowed(cls) -> "OrderPermissionState":
        return cls(
            allow_new_entries=True,
            allow_protection_modify=True,
            allow_normal_close=True,
            allow_emergency_close=True,
        )

    @classmethod
    def local_analysis(cls) -> "OrderPermissionState":
        return cls(
            allow_new_entries=True,
            allow_protection_modify=True,
            allow_normal_close=True,
            allow_emergency_close=False,
        )

    @classmethod
    def from_local_analysis(cls) -> "OrderPermissionState":
        return cls.local_analysis()

    @classmethod
    def live_gate(cls, is_open: bool) -> "OrderPermissionState":
        return cls(
            allow_new_entries=bool(is_open),
            allow_protection_modify=True,
            allow_normal_close=True,
            allow_emergency_close=True,
        )

    @classmethod
    def from_live_gate(cls, is_open: bool) -> "OrderPermissionState":
        return cls.live_gate(is_open)

    @classmethod
    def new_entries_blocked(
        cls,
        *,
        allow_protection_modify: bool = True,
        allow_normal_close: bool = True,
        allow_emergency_close: bool = True,
    ) -> "OrderPermissionState":
        return cls(
            allow_new_entries=False,
            allow_protection_modify=bool(allow_protection_modify),
            allow_normal_close=bool(allow_normal_close),
            allow_emergency_close=bool(allow_emergency_close),
        )

    @classmethod
    def from_daily_bleed_guard(cls, block_reason: Optional[str]) -> "OrderPermissionState":
        if block_reason is None:
            return cls.all_allowed()
        return cls.new_entries_blocked(
            allow_protection_modify=True,
            allow_normal_close=True,
            allow_emergency_close=True,
        )

    def with_new_entries(self, allowed: bool) -> "OrderPermissionState":
        return OrderPermissionState(
            allow_new_entries=bool(allowed),
            allow_protection_modify=self.allow_protection_modify,
            allow_normal_close=self.allow_normal_close,
            allow_emergency_close=self.allow_emergency_close,
        )

    def without_emergency_close(self) -> "OrderPermissionState":
        return OrderPermissionState(
            allow_new_entries=self.allow_new_entries,
            allow_protection_modify=self.allow_protection_modify,
            allow_normal_close=self.allow_normal_close,
            allow_emergency_close=False,
        )

    def combine(self, other: "OrderPermissionState") -> "OrderPermissionState":
        return OrderPermissionState(
            allow_new_entries=self.allow_new_entries and other.allow_new_entries,
            allow_protection_modify=self.allow_protection_modify and other.allow_protection_modify,
            allow_normal_close=self.allow_normal_close and other.allow_normal_close,
            allow_emergency_close=self.allow_emergency_close and other.allow_emergency_close,
        )
