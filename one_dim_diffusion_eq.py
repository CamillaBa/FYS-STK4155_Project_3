import autograd.numpy as np

class one_dim_diff_eq:
    """ One dimensional diffusion equation class.
    An instance of this class represents an 
    approximate solution u(x,t) to the equation 
    u_xx = u_t at a fixed time t.

    The class function "explicit_euler_update" updates
    u(x,t0) to u(x,t1), where t1-t0 = dt using an
    explicit Euler scheme.
    """

    def __init__(self, u, dx, dt):
        """ To initialize an instance, give
        u, representing an initial condition,
        and dx and dt representing the step sizes for
        x and t respectively.
        """
        self.u = u
        self.n = np.size(u);
        self.alpha = dt/dx**2
        self.beta = 1-2*self.alpha
        if self.alpha > 0.5: raise Exception("Using unstable settings for dx and dt.")

    def explicit_euler_update(self):
        """ Update using an explicit Euler scheme.
        """
        u_temp = np.copy(self.u)
        for i in range(1,self.n-1):
            self.u[i]=self.alpha * u_temp[i - 1] + self.beta* u_temp[i] + self.alpha * u_temp[i + 1]