import pytest
import torch

from benchmarks.train_loop import (
    _bootstrap_speedup_ci,
    _close_training,
    _trajectory,
    make_batches,
    step_master,
)


def test_training_policy_checks_loss_dtype_and_all_outputs():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cpu"), 1, seed=17)
    master = tuple(p.float() for p in params)
    expected = step_master(master, *batches[0], 1e-3)
    bad_loss = (expected[0].bfloat16(), expected[1], expected[2])
    with pytest.raises(AssertionError):
        _close_training(bad_loss, expected)
    bad_gradients = (expected[0], tuple(g for g in expected[1][:-1]), expected[2])
    with pytest.raises(AssertionError):
        _close_training(bad_gradients, expected)


def test_teacher_targets_produce_finite_long_trajectory():
    params, batches = make_batches(4, 8, 16, 4, torch.bfloat16, torch.device("cpu"), 4, seed=211)
    master = tuple(p.float() for p in params)
    losses, _ = _trajectory(lambda p, x, y: step_master(p, x, y, 1e-3), master, batches, 100)
    assert len(losses) == 100
    assert all(torch.isfinite(torch.tensor(losses)))


def test_bootstrap_is_deterministic_and_lower_ratio_is_faster():
    a = [1.0, 1.1, 0.9, 1.05]
    b = [2.0, 2.1, 1.9, 2.05]
    first = _bootstrap_speedup_ci(a, b)
    second = _bootstrap_speedup_ci(a, b)
    assert first == second
    assert first["median_ratio"] < 1.0
    assert first["draws"] == 2000 and first["seed"] == 0


def test_report_mode_continues_after_baseline_mismatch():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cpu"), 1, seed=307)
    master = tuple(p.float() for p in params)
    def bad(p, x, y):
        loss, grads, updated = step_master(p, x, y, 1e-3)
        return loss + 1.0, grads, updated
    losses, _, report = _trajectory(bad, master, batches, 4, enforce=False, report=True)
    assert len(losses) == 4
    assert report["passed"] is False
    assert report["first_failure_step"] == 0
    with pytest.raises(AssertionError):
        _trajectory(bad, master, batches, 4)


def test_report_mode_measures_nonzero_error_for_passing_perturbation():
    params, batches = make_batches(2, 3, 4, 2, torch.bfloat16, torch.device("cpu"), 1, seed=101)
    master = tuple(p.float() for p in params)
    def near(p, x, y):
        loss, grads, updated = step_master(p, x, y, 1e-3)
        return loss, grads, tuple(value + 1e-8 for value in updated)
    _, _, report = _trajectory(near, master, batches, 2, enforce=False, report=True)
    assert report["passed"] is True
    assert report["max_absolute_error"] > 0.0
