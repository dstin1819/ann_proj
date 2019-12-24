import numpy as np

class dense:
    def __init__(self, in_size, out_size, init_sigma, with_bias, activation):
        self.in_size = in_size
        self.out_size = out_size
        self.init = init_sigma
        self.activation = activation
        self.output = np.ones([self.out_size,1])

        self.weights = np.reshape(np.array([np.random.normal(0, self.init, self.in_size*self.out_size)]), [self.out_size, self.in_size])
        self.bias = np.reshape(np.array([np.random.normal(0, self.init, self.out_size)]), [self.out_size, 1])

    def forward(self, in_vector):
        self.output = np.dot(self.weights, in_vector) + self.bias
        
        if self.activation == "sigmoid":
            self.output = 1/(1+np.exp(-self.output))
        elif self.activation == "tanh":
            self.output = np.tanh(self.output)
        elif self.activation == "relu":
           self.output[self.output < 0] = 0
        elif self.activation == "softmax":
            self.output = np.exp(self.output)/np.sum(np.exp(self.output))
        else:
            print("Not a valid activation function")
        




test_layer = dense(8, 3, 0.1, False, "softmax")


a = np.ones([8,1]) * 2
test_layer.forward(a)
print(test_layer.output)


