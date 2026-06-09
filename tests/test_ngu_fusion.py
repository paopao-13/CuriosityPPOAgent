"""NGUFusion 融合模块测试, 使用 Mock icm/rnd/episodic."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from curiosity_ppo.config import Config
from curiosity_ppo.curiosity.ngu_fusion import NGUFusion


def _make_config(icm=True, rnd=True, episodic=True, L=5):
    """构建可开关各子模块的 Config."""
    cfg = Config()
    cfg.icm.enabled = icm
    cfg.rnd.enabled = rnd
    cfg.episodic.enabled = episodic
    cfg.episodic.L = L
    return cfg


def _make_mocks():
    """构建 MagicMock 的 icm/rnd/episodic, 返回已知常量."""
    icm = MagicMock()
    icm.compute_reward.return_value = 1.0  # r_icm

    rnd = MagicMock()
    rnd.compute_reward.return_value = 2.0  # r_ngu (rnd-only path)
    rnd.compute_alpha.return_value = 3.0   # alpha_t

    episodic = MagicMock()
    episodic.compute_reward.return_value = 4.0  # r_epi

    return icm, rnd, episodic


def test_full_mode():
    """full: r_icm + r_epi * min(max(alpha, 1), L)."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=True, rnd=True, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    # alpha=3, min(max(3,1),5)=3, r_ngu = 4.0 * 3 = 12.0
    expected = 1.0 + 4.0 * 3.0
    assert np.isclose(r, expected), f"full mode expected {expected}, got {r}"
    icm.compute_reward.assert_called_once()
    episodic.compute_reward.assert_called_once_with('emb')
    rnd.compute_alpha.assert_called_once_with('s_next')


def test_no_icm_mode():
    """no_icm: r_icm=0, 仅 episodic+rnd 融合."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=False, rnd=True, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    expected = 0.0 + 4.0 * 3.0
    assert np.isclose(r, expected), f"no_icm expected {expected}, got {r}"
    icm.compute_reward.assert_not_called()


def test_no_episodic_mode():
    """no_episodic: r_icm + r_ngu(rnd-only path)."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=True, rnd=True, episodic=False, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    # episodic disabled -> r_ngu = rnd.compute_reward(s_next) = 2.0
    expected = 1.0 + 2.0
    assert np.isclose(r, expected), f"no_episodic expected {expected}, got {r}"
    episodic.compute_reward.assert_not_called()
    rnd.compute_reward.assert_called_once_with('s_next')
    rnd.compute_alpha.assert_not_called()


def test_no_rnd_mode():
    """no_rnd: r_icm + r_epi * 1.0 (alpha=1)."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=True, rnd=False, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    # rnd disabled -> alpha=1.0, r_ngu = r_epi * min(max(1,1),5) = 4.0 * 1.0
    expected = 1.0 + 4.0 * 1.0
    assert np.isclose(r, expected), f"no_rnd expected {expected}, got {r}"
    rnd.compute_alpha.assert_not_called()
    rnd.compute_reward.assert_not_called()


def test_all_disabled():
    """所有子模块关闭 -> 返回 0."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=False, rnd=False, episodic=False, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    assert np.isclose(r, 0.0), f"all disabled expected 0.0, got {r}"


def test_alpha_clipped_to_L():
    """alpha > L 时应被裁剪到 L."""
    icm, rnd, episodic = _make_mocks()
    rnd.compute_alpha.return_value = 100.0  # 远大于 L
    cfg = _make_config(icm=False, rnd=True, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    # alpha=100 -> min(max(100,1),5)=5, r_ngu = 4.0 * 5 = 20.0
    expected = 4.0 * 5.0
    assert np.isclose(r, expected), f"alpha clip expected {expected}, got {r}"


def test_alpha_clipped_to_one():
    """alpha < 1 时应被裁剪到 1."""
    icm, rnd, episodic = _make_mocks()
    rnd.compute_alpha.return_value = 0.2  # < 1
    cfg = _make_config(icm=False, rnd=True, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb='emb')
    # alpha=0.2 -> min(max(0.2,1),5)=1, r_ngu = 4.0 * 1.0 = 4.0
    expected = 4.0 * 1.0
    assert np.isclose(r, expected), f"alpha floor expected {expected}, got {r}"


def test_no_controllable_emb_skips_episodic():
    """controllable_emb=None 时即使 episodic 启用也不计算 r_epi."""
    icm, rnd, episodic = _make_mocks()
    cfg = _make_config(icm=True, rnd=True, episodic=True, L=5)
    fusion = NGUFusion(cfg, icm=icm, rnd=rnd, episodic=episodic)

    r = fusion.compute(s_t='s', a='a', s_next='s_next', controllable_emb=None)
    # episodic skipped (emb None), 走 rnd-only 分支
    expected = 1.0 + 2.0
    assert np.isclose(r, expected), f"expected {expected}, got {r}"
    episodic.compute_reward.assert_not_called()
