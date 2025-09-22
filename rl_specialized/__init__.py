"""专用模型强化学习算法包

注意：为避免在未安装可选依赖（如 gymnasium、torch、stable-baselines3）时出现导入错误，
此处不在包级别导入子模块。请按需从具体子包导入，例如：

    from rl_specialized.action_spaces import get_2_player_action_space
    from rl_specialized.utils import create_state_encoder
    from rl_specialized.training.train_specialized import SpecializedTrainer

"""

__version__ = "1.0.0"
