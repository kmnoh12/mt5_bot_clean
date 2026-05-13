from core.optimization.search_space_v4 import flatten_config, grid_trial_configs, random_trial_configs


def test_random_trial_configs_are_seed_reproducible() -> None:
    first = random_trial_configs(5, seed=20260513)
    second = random_trial_configs(5, seed=20260513)

    assert first == second
    assert len(first) == 5
    assert "entry" in first[0]
    assert "risk" in first[0]


def test_grid_trial_configs_honors_limit_and_flattens() -> None:
    configs = grid_trial_configs(3)
    flat = flatten_config(configs[0])

    assert len(configs) == 3
    assert "entry.min_signal_score" in flat
    assert "risk.hard_max_net_loss_usd" in flat

