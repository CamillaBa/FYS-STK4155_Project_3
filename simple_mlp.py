import autograd.numpy as np

class simple_MLP:
    """ A simple MLP class representing a sequence of linear transformations
    with activation functions in between. If network_size = [1,3,4,1], then
    the function input_to_output determines a function R^1->R^1 that factors as
    R^1-> R^3 -> R^4 -> R^1, where each arrow is a linear transformation followed by
    an activation function.

    The choice of activation functions is given as A. If A is a single funtcion,
    then the activation functions through the network are all identical and equal to A.
    If A is a list [A1, A2,.. An] matching network_size = [s1,...,sn,sn+1], then
    the list A determines the succesive activation functions throughout the network in the
    sense that A1 is the first activation function, A2 is the second, and so on.
    """
    def __init__(self, network_size, A):
        # dimensions of (succesive) weight matrices
        L = len(network_size)

        # initiate hidden layers and output layer
        layer_data   = [np.random.randn(network_size[l],network_size[l-1]+1) for l in range(1,L)]
        
        # self objects (in alphabetic order)
        if type(A) == list: self.A = A
        else:               self.A = [A]*(L-1) 
        self.network_size          = network_size
        self.L                     = L
        self.layer_data            = layer_data

    def input_to_output(self, input, layer_data):
        # relabel self variables
        A = self.A
        L = self.L

        # feed forward
        a = input
        for l in range(0,L-1):
            z = np.dot(layer_data[l],np.concatenate((np.array([1]),a)))
            a = A[l](z)

        # return ouput layer
        return a[0]