class Robot():
    def __init__(self, pos, env):
        self.pos = pos
        self.env = env

        self.received_robots = []
        self.received_targets = []
