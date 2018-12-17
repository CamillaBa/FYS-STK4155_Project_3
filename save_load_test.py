#======================================================================================================
# setting up differential equation
#======================================================================================================

# importing classes
from simple_mlp import *

# importing various functions
from save_load_to_file import *
from gradient_descent_methods import *

# importing tools
from timeit import default_timer as timer
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

# various activation functions
def  sigmoid(z): return 1/(1 + np.exp(-z))
def identity(z): return z

network_size = [2,128,128,128,1]
A            = [sigmoid,sigmoid,sigmoid,identity]
network      = simple_MLP(network_size,A)
layer_data   = network.layer_data
L            = network.L

# trial solution
def v(x,t,layer_data):
    input = np.array([x,t])
    return np.sin(np.pi*x) + x * (x-1) * t * network.input_to_output(input,layer_data)

# applying the operator D := Dxx - Dt to the trial solution
v_xx = grad(grad(v,0),0)
v_t  = grad(v,1)
def Dv(x,t,layer_data): return v_xx(x, t, layer_data)-v_t(x, t, layer_data)

# cost function
def cost_function(domain,layer_data):
    x,t     = domain[0], domain[1]
    Dv_eval = np.array([Dv(x_, t_, layer_data) for t_ in t for x_ in x])
    cost    = np.dot(Dv_eval, Dv_eval)
    return cost/np.size(x)/np.size(x)

#======================================================================================================
# starting test 
#
# Train for 3 iterations and save cost, and compare to training 2 iterations,
# saving to file, reopening and training one iteration.
#
#======================================================================================================

x, t        = np.linspace(0,1,11), np.linspace(0,1,11)
domain      = [x,t]


# training and saving cost after 3 iterations
gym1 = momentum_gym(network)
train_momentum(gym1, domain, cost_function,
               iterations      = 3,      
               iteration_start = 1,    # starting iteration
               backup_every    = 100,    # save a backup every number of iterations
               show_prog_every = 1,    # show progress (demanding ~ 1s)
               method          = "gdm",
               folder_name     = "test")

cost1 = cost_function(domain,gym1.network.layer_data)


# initiating network with same seed
np.random.seed(10)
network_size = [2,128,128,128,1]
A            = [sigmoid,sigmoid,sigmoid,identity]
network      = simple_MLP(network_size,A)
layer_data   = network.layer_data
L            = network.L

# training and saving to file after 2 iterations
gym2 = momentum_gym(network)
train_momentum(gym2, domain, cost_function,
               iterations      = 2,      
               iteration_start = 1,    # starting iteration
               backup_every    = 2,    # save a backup every number of iterations
               show_prog_every = 1,    # show progress (demanding ~ 1s)
               method          = "gdm",
               folder_name     = "test")

# load, train 1 iteration, and save cost
network_size_string = "_".join([str(item) for item in network_size])
filename   = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format("test","gdm",network_size_string,2)
gym2 = load_object_from_file(filename)
train_momentum(gym2, domain, cost_function,
               iterations      = 1,      
               iteration_start = 1,    # starting iteration
               backup_every    = 100,    # save a backup every number of iterations
               show_prog_every = 1,    # show progress (demanding ~ 1s)
               method          = "gdm",
               folder_name     = "test")

cost2 = cost_function(domain,gym2.network.layer_data)

print("cost1: ",cost1,", cost2:",cost2, ",   difference: ", cost1-cost2)
if cost1-cost2 == 0: print("Unit test succesful!")
