"""NGU 融合: 将 ICM / RND / episodic 内在奖励按 NGU 公式融合.

融合公式:
    r_int = r_icm + r_ngu

其中:
- r_icm = eta * ICM_forward_loss  (短时程好奇心, 仅当 config.icm.enabled).
- r_ngu:
    - 若 episodic 启用且有 controllable_emb:
        r_ngu = r_episodic * min(max(alpha_t, 1), L)
        alpha_t 由 RND 误差经 sigmoid 映射到 [1, L] (长期调制).
    - 否则若 RND 启用:
        r_ngu = r_rnd  (仅 RND 内在奖励).
    - 否则: 0.
"""
class NGUFusion:
    """NGU 融合模块.

    Args:
        config: Config 实例, 通过 config.icm.enabled / config.rnd.enabled /
            config.episodic.enabled / config.episodic.L 控制融合行为.
        icm: 可选的 ICMCuriosity 实例.
        rnd: 可选的 RNDCuriosity 实例.
        episodic: 可选的 EpisodicMemory 实例.
    """

    def __init__(self, config, icm=None, rnd=None, episodic=None):
        self.config = config
        self.icm = icm
        self.rnd = rnd
        self.episodic = episodic

    def compute(self, s_t=None, a=None, s_next=None, controllable_emb=None,
                episodic_override=None) -> float:
        """计算融合后的内在奖励.

        Args:
            s_t: 当前观测 (传给 ICM).
            a: 动作 (传给 ICM).
            s_next: 下一观测 (传给 ICM / RND).
            controllable_emb: 可控性嵌入 (传给 episodic).
            episodic_override: 可选的 EpisodicMemory 实例, 用于多环境隔离.
                若为 None 则使用构造时传入的 self.episodic.

        Returns:
            float 融合内在奖励.
        """
        episodic = episodic_override if episodic_override is not None else self.episodic

        # ICM 前向损失好奇心 (短时程)
        if self.config.icm.enabled and self.icm and s_t is not None:
            r_icm = self.icm.compute_reward(s_t, a, s_next)
        else:
            r_icm = 0.0

        # Episodic + RND 融合 (NGU)
        if self.config.episodic.enabled and episodic and controllable_emb is not None:
            r_epi = episodic.compute_reward(controllable_emb)
            if self.config.rnd.enabled and self.rnd and s_next is not None:
                alpha = self.rnd.compute_alpha(s_next)
            else:
                alpha = 1.0
            r_ngu = r_epi * min(max(alpha, 1.0), float(self.config.episodic.L))
        elif self.config.rnd.enabled and self.rnd and s_next is not None:
            r_ngu = self.rnd.compute_reward(s_next)
        else:
            r_ngu = 0.0

        return r_icm + r_ngu
