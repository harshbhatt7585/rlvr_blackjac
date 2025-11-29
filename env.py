class BlackjacEnv:
    def __init__(self):
        self.state = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(1,))

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += action
        return self.state, reward, done, info