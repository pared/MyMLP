
class Layer:
    def __init__(self, num_nodes: int, activation: str):
        self.num_nodes = num_nodes
        self.activation = activation

class ModelLayer:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation