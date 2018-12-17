# importing classes
from simple_mlp import *

# importing various functions
from save_load_to_file import *
from gradient_descent_methods import *

# importing tools
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt

#=======================================================================================================
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

#=======================================================================================================
# Training session (takes a long time)
#=======================================================================================================

x, t           = np.linspace(0,1,11), np.linspace(0,1,11)
domain         = [x,t]
iterations     = 10000
eta            = {}
eta["gdm"]     = 0.005
eta["adam"]    = 0.00075

# best parameters that we found (by trial and error)
#gym     = momentum_gym(network)
#gym.eta = eta["adam"]
#train_momentum(gym, domain, cost_function,
#               iterations      = iterations ,      
#               iteration_start = 1,    # starting iteration
#               backup_every    = 5,    # save a backup every number of iterations
#               show_prog_every = 50,   # show progress (demanding ~ 1s)
#               method          = "adam",
#               folder_name     = "long_session")

#==========================================================================================================
# Plotting results of adam/gdm vs iterations (takes a long time)
#==========================================================================================================

#data                = {}
#folder_name         = "long_session"
#network_size_string = "_".join([str(item) for item in network_size])

#plt.figure("cost_vs_iteration")
#for method in ["gdm", "adam"]:
#    data[method] = []
#    iterations   = range(0,5005,5)
#    N            = len(iterations)*5
#    for iteration in iterations:
#        filename   = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format(folder_name,method,network_size_string,iteration)
#        gym        = load_object_from_file(filename)
#        eta        = gym.eta
#        gamma      = gym.gamma
#        beta1      = gym.beta1
#        beta2      = gym.beta2
#        layer_data = gym.network.layer_data
#        data[method].append(cost_function(domain,layer_data))
#        print("method: ", method, " completion: {:.2%}".format(iteration/(N+1)))
#    plotname = method
#    if method != "adadelta": plotname += r", $\eta=${}".format(eta)
#    if method in {"gdm", "nag", "rms", "adadelta"}: plotname += r", $\gamma=${}".format(gamma) 
#    if method == "adam":
#        plotname += r", $\beta_1=${}".format(beta1) 
#        plotname += r", $\beta_2=${}".format(beta2) 
#    plt.plot(iterations, data[method], label = plotname)

#plt.legend(loc="best")
#plt.xlabel("iterations")
#plt.ylabel("cost function")
#plt.yscale('log')
#plt.title("1-dimensional diffusion equation \n MLP size = {}".format(network_size))

#==========================================================================================================
# Plotting results of best method vs analytical solution
#==========================================================================================================

def abserror(target, approximation):
    return np.linalg.norm(target-approximation)

def color_style_analytical(t):
    if t == "t1": return "lightpink"
    if t == "t2": return "deepskyblue"

def color_style(t):
    if t == "t1": return "darkred"
    if t == "t2": return "blue"

timelist      = {}
timelist["t1"]=0.05
timelist["t2"]=1

def analytical(x,t): return np.sin(np.pi*x)*np.exp(-np.pi**2*t)

def plot_network_from_file(method,iterations,network_size, x, t, folder_name= "default"):
    network_size_string = "_".join([str(item) for item in network_size])
    filename   = "./data/{}/{}/network_size_{}_iteration_{}.pickle".format(folder_name,method,network_size_string,iterations)
    layer_data = load_object_from_file(filename).network.layer_data
    dx = x[1]-x[0]
    plt.figure("neural_network_heat_eq_dx_{}".format(dx))
    plt.title("MLP size = {}\n iterations = {}, dx = {}".format(network_size,iterations,dx))
    for time in ["t1","t2"]:
        u = np.array([v(x_,timelist[time],layer_data) for x_ in x])
        plt.plot(x,u,
                 color = color_style(time),
                 label = r"$t=$"+str(timelist[time])+", abs err: " + str(abserror(analytical(x,timelist[time]),u)))
        plt.plot(x,analytical(x,timelist[time]), 
                 color = color_style_analytical(time), 
                 label = r"$t=$"+str(timelist[time])+" (analytical)",
                 linestyle = '--')

    plt.xlabel(r"$x$",fontsize = 14)
    plt.ylabel(r"$u$",fontsize = 14)
    plt.legend(loc="best")
    plt.show(block=False)
    
    # print cost function
    print("iteration: ",iterations, "  Cost function: ", cost_function([x,t],layer_data))

# adam (best)

#x, t = np.linspace(0,1,11), np.linspace(0,1,11)
#plot_network_from_file("adam",4840,network_size,x,t,folder_name = "long_session")
#x, t = np.linspace(0,1,101), np.linspace(0,1,11)
#plot_network_from_file("adam",4840,network_size,x,t,folder_name = "long_session")
#plt.show()


# gdm (best)

x, t = np.linspace(0,1,11), np.linspace(0,1,11)
plot_network_from_file("gdm",iterations,network_size,x,t,folder_name = "long_session")
x, t = np.linspace(0,1,101), np.linspace(0,1,11)
plot_network_from_file("gdm",iterations,network_size,x,t,folder_name = "long_session")
plt.show()

#=======================================================================================================
# Success message
#=======================================================================================================

plt.show()
print("Success!")