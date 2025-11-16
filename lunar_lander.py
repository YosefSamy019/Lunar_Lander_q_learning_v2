import gymnasium as gym
import numpy as np

class LunarLanderEnv:
    def __init__(self):
        self._env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self._last_state = None
        self.reset()

    # --------------------------------------------------
    #  Info
    # --------------------------------------------------
    def getStatesActionsNames(self):
        # State: 8 continuous values
        state_names = [
            "state x_pos", "state y_pos",
            "state x_vel", "state y_vel",
            "state angle", "state angular_vel",
            "state left_leg_contact",
            "state right_leg_contact"
        ]

        # Actions: 0,1,2,3 (fire nothing / left / main / right)
        action_names = [
            "action no_op",
            "action fire_left",
            "action fire_main",
            "action fire_right"
        ]

        return state_names, action_names

    def getStatesMinMax(self):
        # Gym provides bounds (many are inf, so we clip)
        low  = self._env.observation_space.low
        high = self._env.observation_space.high

        # Replace infinite bounds with large finite numbers
        low  = np.where(np.isinf(low),  -5.0, low)
        high = np.where(np.isinf(high),  5.0, high)

        return low.astype(np.float32), high.astype(np.float32)

    def _getCurrentState(self):
        return np.array(self._last_state, dtype=np.float32)

    def getStateShape(self):
        return self._getCurrentState().shape

    def getActionsCount(self):
        return self._env.action_space.n    # always 4

    # --------------------------------------------------
    #  Reset
    # --------------------------------------------------
    def reset(self):
        s, _ = self._env.reset()
        self._last_state = s
        return self._getCurrentState()

    # --------------------------------------------------
    #  Rendering
    # --------------------------------------------------
    def render(self):
        return self._env.render()

    # --------------------------------------------------
    #  Single step
    # --------------------------------------------------
    def step(self, action):
        """
        Returns:
        next_state, done, reward
        """
        s, r, terminated, truncated, _ = self._env.step(int(action))
        self._last_state = s
        done = terminated or truncated
        return self._getCurrentState(), done, r