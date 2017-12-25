from layer import Layer
from my_mulitlayer_perceptron import MyMLP

mlp = MyMLP(2,
            [Layer(3, 'sigmoid'),
             Layer(1, 'sigmoid')])
mlp.print_weights()
mlp.summary()