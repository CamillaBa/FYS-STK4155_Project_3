# importing classes
from one_dim_diffusion_eq import *

# importing various functions
from save_load_to_file import *

# importing tools
import autograd.numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# calculating data
#===============================================================================

#for dx in [0.1,0.01]:
#    dt = 0.01*dx*dx
#    x = np.arange(0,1+dx,dx)
#    u_init = np.sin(np.pi*x)

#    # initiating one_dim_diff_eq instance
#    eq = one_dim_diff_eq(u_init,dx,dt)

#    # save solution at t=0.05
#    i = 0
#    while (i*dt <= 0.05):
#        eq.explicit_euler_update()
#        i+=1
#    save_object_to_file(eq.u, "./data/explicit_euler_dx_"+str(dx)+"_t1.pickle")

#    # save solution at t=1.00
#    i = 0
#    while (i*dt <= 0.95):
#        eq.explicit_euler_update()
#        i+=1
#    save_object_to_file(eq.u, "./data/explicit_euler_dx_"+str(dx)+"_t2.pickle")

#===========================================================================
# Plot formatting
#===========================================================================

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

N = 100 # terms of analytical solution

def A(n):
    # fourier coefficients of initial condition g(x)
    if n == 1: return 1
    else: return 0

An = np.array([A(n) for n in range(1,N+1)])

def analytical(x,t):
    """ Analytical solution to the diffusion equation with the given boundary conditions.
    An represents the fourier coefficients of the initial condition g(x).
    """
    pi_x           = np.pi*x
    mat_1          = np.array([np.sin(n*pi_x) for n in range(1,N+1)])
    exponentials   = np.array([np.exp(-n**2*np.pi**2*t) for n in range(1,N+1)])
    mat_2          = An*exponentials
    multiplication = np.matmul(mat_2,mat_1)
    return  multiplication

for dx in ["0.1","0.01"]:
    x = np.arange(0,1+float(dx),float(dx))
    for method in ["explicit_euler"]:
        plt.figure(method+"_dx_"+dx)
        plt.title(method+"\n"+r" $dx=$"+dx+r" $dt=$"+"0.01"+r"$dx^2$")
        for t in ["t1","t2"]:
            u = load_object_from_file("./data/"+method+"_dx_"+dx+"_"+t+".pickle")
            plt.plot(x,u,
                     color = color_style(t),
                     label = r"$t=$"+str(timelist[t])+", abs err: " + str(abserror(analytical(x,timelist[t]),u)))
            plt.plot(x,analytical(x,timelist[t]), 
                     color = color_style_analytical(t), 
                     label = "$t=$"+str(timelist[t])+" (analytical)",
                     linestyle = '--')
        plt.xlabel(r"$x$",fontsize = 14)
        plt.ylabel(r"$u$",fontsize = 14)
        plt.legend(loc="best")
        plt.show(block=False)

#===========================================================================
# Success
#===========================================================================

print("Success!")
plt.show()