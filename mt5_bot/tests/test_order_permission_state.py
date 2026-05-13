import unittest

from execution.order_permissions import OrderPermissionState


class OrderPermissionStateTests(unittest.TestCase):
    def test_all_allowed_enables_every_permission(self) -> None:
        state = OrderPermissionState.all_allowed()
        self.assertTrue(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)
        self.assertTrue(state.allow_normal_close)
        self.assertTrue(state.allow_emergency_close)

    def test_new_entries_blocked_keeps_protection_modify_allowed_by_default(self) -> None:
        state = OrderPermissionState.new_entries_blocked()
        self.assertFalse(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)

    def test_new_entries_blocked_keeps_normal_close_allowed_by_default(self) -> None:
        self.assertTrue(OrderPermissionState.new_entries_blocked().allow_normal_close)

    def test_new_entries_blocked_keeps_emergency_close_allowed_by_default(self) -> None:
        self.assertTrue(OrderPermissionState.new_entries_blocked().allow_emergency_close)

    def test_emergency_close_does_not_leak_from_new_entry_permission(self) -> None:
        state = OrderPermissionState(
            allow_new_entries=True,
            allow_protection_modify=True,
            allow_normal_close=True,
            allow_emergency_close=False,
        )
        self.assertTrue(state.allow_new_entries)
        self.assertFalse(state.allow_emergency_close)

    def test_live_gate_closed_blocks_new_entries(self) -> None:
        self.assertFalse(OrderPermissionState.live_gate(False).allow_new_entries)

    def test_live_gate_closed_still_allows_protection_modify(self) -> None:
        self.assertTrue(OrderPermissionState.live_gate(False).allow_protection_modify)

    def test_live_gate_closed_still_allows_closes(self) -> None:
        state = OrderPermissionState.live_gate(False)
        self.assertTrue(state.allow_normal_close)
        self.assertTrue(state.allow_emergency_close)

    def test_live_gate_open_allows_new_entries(self) -> None:
        self.assertTrue(OrderPermissionState.live_gate(True).allow_new_entries)

    def test_local_analysis_does_not_enable_emergency_close(self) -> None:
        state = OrderPermissionState.local_analysis()
        self.assertTrue(state.allow_new_entries)
        self.assertFalse(state.allow_emergency_close)

    def test_daily_bleed_block_maps_to_new_entry_only_block(self) -> None:
        state = OrderPermissionState.from_daily_bleed_guard("DAILY_BLEED_NET_LOSS_LIMIT")
        self.assertFalse(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)
        self.assertTrue(state.allow_normal_close)
        self.assertTrue(state.allow_emergency_close)

    def test_daily_bleed_clear_maps_to_all_allowed(self) -> None:
        self.assertEqual(OrderPermissionState.from_daily_bleed_guard(None), OrderPermissionState.all_allowed())

    def test_combine_uses_most_restrictive_permissions(self) -> None:
        state = OrderPermissionState.live_gate(False).combine(OrderPermissionState.local_analysis())
        self.assertFalse(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)
        self.assertTrue(state.allow_normal_close)
        self.assertFalse(state.allow_emergency_close)

    def test_with_new_entries_does_not_change_close_permissions(self) -> None:
        state = OrderPermissionState.local_analysis().with_new_entries(False)
        self.assertFalse(state.allow_new_entries)
        self.assertTrue(state.allow_protection_modify)
        self.assertTrue(state.allow_normal_close)
        self.assertFalse(state.allow_emergency_close)

    def test_without_emergency_close_does_not_change_new_entries(self) -> None:
        state = OrderPermissionState.all_allowed().without_emergency_close()
        self.assertTrue(state.allow_new_entries)
        self.assertFalse(state.allow_emergency_close)


if __name__ == "__main__":
    unittest.main()
