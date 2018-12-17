# importing classes
from one_dim_diffusion_eq import *
from simple_mlp import *

# importing various functions
from save_load_to_file import *

# importing tools
from timeit import default_timer as timer
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt

class momentum_gym:
    """ An instance of momentum gym represents a training
    session for a given MLP. As constructor argument, we give thus
    give an MLP.

    The class functions gdm, nag, rms and adam implement various
    momentum based gradient decent learning algorithms. The class function
    gd implements regular gradient descent
    """
    def __init__(self, network):
        self.network = network
        self.L       = network.L

        # training parameters
        self.gamma   = 0.9   # momentum parameter
        self.eta     = 0.001 # learning rate
        self.beta1   = 0.9
        self.beta2   = 0.99
        self.epsilon = 1e-8

        # default momentums
        self.v = [0.0]*self.L
        self.m = [0.0]*self.L

    def gd(self, domain, cost_function_grad):
        # gradient descent
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):  
            layer_data[l] = layer_data[l] - self.eta*cost_function_grad_eval[l]

    def gdm(self, domain, cost_function_grad):
        # gradient descent with momentum
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):  
            self.v[l]     = self.gamma*self.v[l] - self.eta*cost_function_grad_eval[l]
            layer_data[l] = layer_data[l] + self.v[l]

    def nag(self, domain, cost_function_grad):
        # Nesterov accelerated gradient
        layer_data              = self.network.layer_data
        expected_layer_data     = [layer_data[l] + self.gamma*self.v[l] for l in range(0,self.L-1)]
        cost_function_grad_eval = cost_function_grad(domain,expected_layer_data)
        for l in range(0,self.L-1):
            self.v[l]     = self.gamma*self.v[l] - self.eta*cost_function_grad_eval[l]
            layer_data[l] = layer_data[l] + self.v[l]

    def rms(self, domain, cost_function_grad):
        # root mean square propagation
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):
             self.v[l]     =  self.gamma*self.v[l] + (1-self.gamma)*cost_function_grad_eval[l]**2
             layer_data[l] =  layer_data[l] - self.eta/np.sqrt(self.v[l]+self.epsilon)*cost_function_grad_eval[l]

    def adam(self, domain, cost_function_grad,iteration):
        # adaptive movement estimation
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):
            self.m[l]     = self.beta1 * self.m[l] +(1-self.beta1)* cost_function_grad_eval[l]
            self.v[l]     = self.beta2 * self.v[l] +(1-self.beta2)* cost_function_grad_eval[l]**2
            m             = self.m[l]/(1-self.beta1**iteration)
            v             = self.v[l]/(1-self.beta2**iteration)
            layer_data[l] =  layer_data[l] - self.eta*m/(np.sqrt(v)+self.epsilon)

    def adagrad(self, domain, cost_function_grad):
        # adaptive gradient algorithm
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):
            self.v[l]     = self.v[l] + cost_function_grad_eval[l]**2
            layer_data[l] = layer_data[l] - self.eta*cost_function_grad_eval[l]/np.sqrt(self.v[l]+self.epsilon)

    def adadelta(self, domain, cost_function_grad):
        # adaptive gradient algorithm
        layer_data              = self.network.layer_data
        cost_function_grad_eval = cost_function_grad(domain,layer_data)
        for l in range(0,self.L-1):
            self.v[l]     = self.v[l]*self.gamma + (1-self.gamma)*cost_function_grad_eval[l]**2
            m             = np.sqrt((self.m[l]+self.epsilon)/(self.v[l]+self.epsilon))*cost_function_grad_eval[l]
            layer_data[l] = layer_data[l] - m
            self.m[l]     = self.gamma*self.m[l]+(1-self.gamma)*m*m


def train_momentum(gym, domain, cost_function,
                   iterations,      
                   iteration_start = 1,   # starting iteration
                   backup_every    = 100, # saves a backup every number of iterations
                   show_prog_every = 10,  # shows progress (demanding)
                   method          = "gdm",
                   folder_name     = "default"):
 
    cost_function_grad  = grad(cost_function,1)
    network_size        = gym.network.network_size
    network_size_string = "_".join([str(item) for item in network_size])

    # save initial settings
    filename = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format(folder_name,method,network_size_string,0)
    save_object_to_file(gym,filename)

    print("Initial cost: ", cost_function(domain,gym.network.layer_data))
    for iteration in range(iteration_start,iteration_start+iterations):
        if method == "gd"      : gym.gd(domain,cost_function_grad)
        if method == "gdm"     : gym.gdm(domain,cost_function_grad)
        if method == "nag"     : gym.nag(domain,cost_function_grad)
        if method == "rms"     : gym.rms(domain,cost_function_grad)
        if method == "adam"    : gym.adam(domain,cost_function_grad,iteration)
        if method == "adagrad" : gym.adagrad(domain,cost_function_grad)
        if method == "adadelta": gym.adadelta(domain,cost_function_grad)

        # print progress
        if iteration % show_prog_every == 0:
            print("Completed iteration: ", iteration, "  Cost function: ", cost_function(domain,gym.network.layer_data))

        # save backup
        if iteration % backup_every == 0:
            filename = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format(folder_name,method,network_size_string,iteration)
            save_object_to_file(gym,filename)
