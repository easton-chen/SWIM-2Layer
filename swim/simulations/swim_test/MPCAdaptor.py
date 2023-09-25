# adapt weights, control bounds
class MPCAdaptor:
    def __init__(self):
        self.t = 0

    def adapt(self, t):
        weights = [0.33, 0.33, 0.33]
        bounds1 = [0, 1]
        bounds2 = [1, 3]
        return weights, bounds1, bounds2
