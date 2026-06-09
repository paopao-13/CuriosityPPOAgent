"""测试全局种子设置: random / numpy / torch 三方均可复现."""
import random

import numpy as np
import torch

from curiosity_ppo.utils.seed import set_seed


def test_set_seed_makes_random_reproducible():
    set_seed(42)
    a = random.random()
    b = random.random()

    set_seed(42)
    c = random.random()
    d = random.random()

    assert a == c, f"random not reproducible: {a} != {c}"
    assert b == d, f"random not reproducible: {b} != {d}"


def test_set_seed_makes_numpy_reproducible():
    set_seed(42)
    a = np.random.random(5)

    set_seed(42)
    b = np.random.random(5)

    assert np.array_equal(a, b), f"numpy not reproducible:\n{a}\n{b}"


def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    a = torch.randn(3, 4)

    set_seed(42)
    b = torch.randn(3, 4)

    assert torch.equal(a, b), f"torch not reproducible:\n{a}\n{b}"


def test_different_seeds_give_different_results():
    set_seed(42)
    r1 = random.random()
    n1 = np.random.random()
    t1 = torch.randn(1).item()

    set_seed(123)
    r2 = random.random()
    n2 = np.random.random()
    t2 = torch.randn(1).item()

    assert r1 != r2, "different seeds gave same random value"
    assert n1 != n2, "different seeds gave same numpy value"
    assert t1 != t2, "different seeds gave same torch value"


def test_set_seed_default_arg():
    """set_seed() 不带参数时应正常运行 (默认 42)."""
    set_seed()
    a = torch.randn(2)
    set_seed()
    b = torch.randn(2)
    assert torch.equal(a, b), "default seed not reproducible"
