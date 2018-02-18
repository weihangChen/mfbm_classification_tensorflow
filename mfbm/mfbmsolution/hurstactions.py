class HurstConfig(object):
    def __init__(self, default_hurst, hurst_actions):
        self.default_hurst = default_hurst
        self.hurst_actions = hurst_actions
        self.label = None
        self.alphabet = None



class HurstUpOrDown(object):
    def __init__(self, base_hurst, base_position, n, action):
        self.base_hurst = base_hurst
        self.base_position= base_position
        self.n = n
        self.scale = 0.6
        self.action = action

    def get_hurst(self, current_position):
        increment = (current_position/self.n  - self.base_position/self.n) * self.scale
        if self.action == "down":
            increment = increment * -1
        new_hurst = self.base_hurst + increment
        return new_hurst