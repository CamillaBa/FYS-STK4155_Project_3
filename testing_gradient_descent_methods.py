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

#========================================================================================================
# setting up MLP and trial solution
#=======================================================================================================

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

#==========================================================================================================
# Training session (fixed domain)
#==========================================================================================================

x, t        = np.linspace(0,1,11), np.linspace(0,1,11)
domain      = [x,t]

iterations  = 1005
eta         = {}

# best parameters that we found (by trial and error)
eta["gd"]       = 0.001
eta["gdm"]      = 0.005
eta["nag"]      = 0.003
eta["rms"]      = 0.00075
eta["adam"]     = 0.00075
eta["adagrad"]  = 0.005

#for method in ["adadelta","adagrad","rms"]: #["gd", "gdm", "nag", "rms", "adam","adagrad","adadelta"]
#    # initiate network with same seed
#    np.random.seed(10)
#    network = simple_MLP(network_size,A)
    
#    # set up network gym
#    gym = momentum_gym(network)
#    if method != "adadelta": gym.eta = eta[method] # learning rate parameter  
#    train_momentum(gym, domain, cost_function,
#                   iterations      = iterations,      
#                   iteration_start = 1,    # starting iteration
#                   backup_every    = 5,    # save a backup every number of iterations
#                   show_prog_every = 50,    # show progress (demanding ~ 1s)
#                   method          = method,
#                   folder_name     = "diffusion_eq")

#==========================================================================================================
# Plotting results of different methods vs iterations
#==========================================================================================================
 
x, t                = np.linspace(0,1,11), np.linspace(0,1,11)
domain              = [x,t]
data                = {}
folder_name         = "diffusion_eq"
network_size_string = "_".join([str(item) for item in network_size])

plt.figure("cost_vs_iteration")
for method in ["gd", "gdm", "nag", "rms", "adam","adagrad","adadelta"]: #["gd", "gdm", "nag", "rms", "adam","adagrad","adadelta"]
    data[method] = []
    iterations   = range(0,1005,50)
    N            = len(iterations)*50
    for iteration in iterations:
        filename   = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format(folder_name,method,network_size_string,iteration)
        gym        = load_object_from_file(filename)
        eta        = gym.eta
        gamma      = gym.gamma
        beta1      = gym.beta1
        beta2      = gym.beta2
        layer_data = gym.network.layer_data
        data[method].append(cost_function(domain,layer_data))
        print("method: ", method, " completion: {:.2%}".format(iteration/(N+1)))
    plotname = method
    if method != "adadelta": plotname += r", $\eta=${}".format(eta)
    if method in {"gdm", "nag", "rms", "adadelta"}: plotname += r", $\gamma=${}".format(gamma) 
    if method == "adam":
        plotname += r", $\beta_1=${}".format(beta1) 
        plotname += r", $\beta_2=${}".format(beta2) 
    plt.plot(iterations, data[method], label = plotname)



plt.legend(loc="best")
plt.xlabel("iterations")
plt.ylabel("cost function")
plt.yscale('log')
plt.title("1-dimensional diffusion equation \n MLP size = {}".format(network_size))
plt.show()

print("Success!")