from .Integrator import Integrator, SingleThreadIntegrator
from .Lyapunov import spectrum_Lyapunov

import numpy as np
from numba import njit, cfunc
from numbalsoda import lsoda_sig


@njit
def h_function(k: int, a: float) -> float:
    """
    Defines the h function that appears in the system of ODEs, see Eq. (17c) in the paper.

    Parameters
    ----------
    k: int
        x Fourier mode

    a: float
        Sum of squared x and y Fourier modes

    Returns
    -------
    float
        Value of the h function for given k and a

    Example
    -------
    >>> import numpy as np
    >>> print(h_function(1, 2 * np.pi))
    0.2
    """
    return ((k - 0.5) ** 2) / ((k - 0.5) ** 2 + (a ** 2) / (4 * np.pi ** 2))

@njit
def g_function(k: int, a: float) -> float:
    """
    Defines the g function that appears in the system of ODEs, see Eq. (17b) in the paper.

    Parameters
    ----------
    k: int
        x Fourier mode

    a: float
        Sum of squared x and y Fourier modes

    Returns
    -------
    g: float
        Value of the g function for given k and a

    Example
    -------
    >>> import numpy as np
    >>> print(g_function(1, 2 * np.pi))
    1.8
    """
    return h_function(k, a) + ((a ** 2) / (2 * np.pi ** 2)) / ((k - 0.5) ** 2 + (a ** 2) / (4 * np.pi ** 2))

@njit
def f_function(k: int, a: float, m: int) -> float:
    """
    Defines the f function that appears in the system of ODEs, see Eq. (17a) in the paper.

    Parameters
    ----------
    k: int
        x Fourier mode

    a: float
        Sum of squared x and y Fourier modes

    m: int
        y Fourier mode

    Returns
    -------
    f: float
        Value of the f function for given k and a

    Example
    -------
    >>> import numpy as np
    >>> print(f_function(1, 2 * np.pi, 1))
    0.07205061947899576
    """
    return ((2 * m ** 2) / np.pi ** 2) * (h_function(k, a) / ((k - 0.5) ** 2 - m ** 2) ** 2)

@njit
def partial_sum_Jacobian_function(X: np.ndarray, kc: int, a: float, m: int) -> float:
    """
    Defines the summation part arising in the Jacobian evaluated at the non-trivial equilibrium points.

    Parameters
    ----------
    X: numpy.ndarray (of shape (kc + 2,))
        Point at which to evaluate the Jacobian matrix

    kc: int
        Cut-off x Fourier mode

    a: float
        Sum of squared x and y Fourier modes

    m: int
        y Fourier mode

    Returns
    -------
    total: float
        Value of the sum

    Example
    -------
    >>> import numpy as np
    >>> kc = 2
    >>> a = np.pi * np.sqrt(2)
    >>> m = 1
    >>> X = np.array([1, 2, 3, 4])
    >>> print(partial_sum_Jacobian_function(X, kc, a, m))
    1.463282581055241
    """
    total = 0

    for i in range(1, kc + 1):
        total += f_function(i, a, m) * (3 * X[0] ** 2 + X[i + 1])

    return total

@njit
def Jacobian_function(X: np.ndarray, kc: int, gamma: float, a: float, m: int) -> np.ndarray:
    """
    Constructs the Jacobian matrix of the Pedlosky ODE system evaluated at the point X.

    Parameters
    ----------
    X: numpy.ndarray (of shape (kc + 2,))
        Point at which to evaluate the Jacobian matrix

    kc: int
        Cut-off x Fourier mode

    gamma: float
        Dissipation parameter

    a: float
        Sum of squared x and y Fourier modes

    m: int
        y Fourier mode

    Returns
    -------
    DF: numpy.ndarray (of shape (kc + 2, kc + 2))
        Jacobian matrix of the system evaluated at the point X

    Example
    -------
    >>> import numpy as np
    >>> kc = 2
    >>> gamma = 0.5
    >>> a = np.pi * np.sqrt(2)
    >>> m = 1
    >>> X = np.array([1, 2, 3, 4])
    >>> print(Jacobian_function(X, kc, gamma, a, m))
    [[-0.5         1.          0.          0.        ]
     [-0.33828258 -0.25       -0.12008437 -0.10611091]
     [ 1.66666667  0.         -0.16666667 -0.        ]
     [ 1.18181818  0.         -0.         -0.40909091]]
    """
    k_max = kc + 2
    DF = np.zeros((k_max, k_max), dtype=np.float64)

    DF[0, 0] = - gamma
    DF[0, 1] = 1.0

    DF[1, 0] = 1.0 + gamma ** 2 / 2 - partial_sum_Jacobian_function(X, kc, a, m)
    DF[1, 1] = - gamma / 2
    for k in range(1, kc + 1):
        DF[1, k + 1] = - f_function(k, a, m) * X[0]

    for k in range(1, kc + 1):
        row = k + 1
        DF[row, row] = - gamma * h_function(k, a) 
        DF[row, 0] = 2.0 * gamma * g_function(k, a) * X[0]

    return DF

@njit
def sum_part(X: np.ndarray, kc: int, a: float, m: int) -> float:
    """
    Defines the summation part (without the coefficients) of the differential equation for B, see the right-hand side of Eq. (16b) in the paper.

    Parameters
    ----------
    X: numpy.ndarray (of shape (kc + 2,))
        Vector of variables

    kc: int
        Cut-off x Fourier mode

    a: float
        Sum of squared x and y Fourier modes

    m: int
        y Fourier mode

    Returns
    -------
    total: float
        Value of the summation part of the differential equation for B

    Example
    -------
    >>> import numpy as np
    >>> kc = 2
    >>> a = np.pi * np.sqrt(2)
    >>> m = 1
    >>> X = np.array([1, 2, 3, 4])
    >>> print(sum_part(X, kc, a, m))
    1.0108920248113646
    """
    total = 0

    for i in range(1, kc + 1):
        total +=  f_function(i, a, m) * (X[0] ** 2 + X[i + 1])

    return total

@cfunc(lsoda_sig)
def system_ODE(t, X: np.ndarray, dX: np.ndarray, p: np.ndarray) -> None:
    """
    Defines the system of ODEs of the dynamical system in terms of the functions defined in the "System" module, see Eqs. (16a)-(16c) in the paper.

    Parameters
    ----------
    t: float
        Time (unused since the system is autonomous)
    
    X: numpy.ndarray (of shape (p[0] + 2,))
        Vector of variables

    dX: numpy.ndarray (of shape (p[0] + 2,))
        Vector of derivatives

    p: numpy.ndarray (of shape (4,))
        Vector of parameters of the form np.array([kc, gamma, a, m])

    Returns
    -------
    None (the function serves as input for the integrator)
    """
    dX[0] = X[1] - p[1] * X[0]
    dX[1] = - (p[1] / 2) * X[1] + ((p[1] ** 2) / 2) * X[0] + X[0] - X[0] * sum_part(X, p[0], p[2], p[3])

    for i in range(1, p[0] + 1):
        dX[i + 1] = p[1] * (g_function(i, p[2]) * X[0] ** 2 - h_function(i, p[2]) * X[i + 1])

@cfunc(lsoda_sig)
def system_tangent(t, X: np.ndarray, dX: np.ndarray, p: np.ndarray) -> None:
    """
    Defines the Pedlosky system and its tangent system at a given point X. In this function, the tangent system is computed by multiplying the Jacobian matrix of the Pedlosky system by the matrix of the tangent system.

    Parameters
    ----------
    t: float
        Time (unused since the system is autonomous)
    
    X: numpy.ndarray (of shape (p[0] + 2,))
        Vector of variables

    dX: numpy.ndarray (of shape (p[0] + 2,))
        Vector of derivatives

    p: numpy.ndarray (of shape (4,))
        Vector of parameters of the form np.array([kc, gamma, a, m])

    Returns
    -------
    None (the function serves as input for the integrator)
    """
    kc = int(p[0]) 
    dim  = kc + 2

    dX[0] = X[1] - p[1] * X[0]
    dX[1] = - (p[1] / 2) * X[1] + ((p[1] ** 2) / 2) * X[0] + X[0] - X[0] * sum_part(X, kc, p[2], p[3])

    for i in range(1, p[0] + 1):
        dX[i + 1] = p[1] * (g_function(i, p[2]) * X[0] ** 2 - h_function(i, p[2]) * X[i + 1])

    var_array = np.empty((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            var_array[i, j] = X[dim * (i + 1) + j]

    tangent_system = Jacobian_function(X, kc, p[1], p[2], p[3]) @ var_array

    k = 0
    for i in range(dim):
        for j in range(dim):
            dX[dim + k] = tangent_system[i][j]
            k += 1

class System:

    def __init__(self, kc: int, gamma: float, a: float, m: int):
        self._kc = kc
        self._gamma = gamma
        self._a = a
        self._m = m

    @property
    def kc(self) -> int:
        return self._kc
    
    @property
    def gamma(self) -> float:  
        return self._gamma
    
    @property
    def a(self) -> float:  
        return self._a
    
    @property
    def m(self) -> int:  
        return self._m

    def h(self, k: int) -> float: 
        return h_function(k, self._a)

    def g(self, k: int) -> float: 
        return g_function(k, self._a)

    def f(self, k: int) -> float: 
        return f_function(k, self._a, self._m)
    
    def transit_function_Phi(self, k: int) -> float:
            """
            Defines the function of the parameters that appears in the sum part of Phi(T, y), see Eq. (19) in the paper.

            Parameters
            ----------
            k: int
                x Fourier mode

            Returns
            ------
            tr_Phi: float
                Value of the transit function for given parameters

            Example
            -------
            >>> ds = System(4, 0.5, np.pi * np.sqrt(2), 1)
            >>> print(ds.transit_function_Phi(1))
            -1.7777777777777777
            """
            tr_Phi = (self._m / (((k - 0.5) ** 2 - self._m ** 2) * ((k - 0.5) ** 2 + self._a ** 2 / (4 * np.pi ** 2)))) 

            return tr_Phi
    
    def Phi(self, y: float, X: np.ndarray) -> float:
        """
        Given the solution of the system of ODEs, computes the value of the function Phi(T, y) at time T as defined in Eq. (19) in the paper.

        Parameters
        ----------
        y: float
            Rescaled horizontal coordinate

        X: numpy.ndarray (of shape (n_ic, len(t_eval), self._kc + 2))
            Solution of the Pedlosky system obtained by integrating the system of ODEs. n_ic can be equal to 1.

        Return
        ------
        phi: float
            Value of the function Phi(T, y) at time T

        Example
        -------
        >>> import numpy as np
        >>> Phi(1, np.array([3, 1, 2 * np.pi, 1]), 1, np.array([1, 2, 3]))
        >>> -0.030776453508032577
        """
        phi = 0

        for k in range(1, self._kc + 1):
            phi += self.transit_function_Phi(k) * (X[0, :, 0] ** 2 + X[0, :, k + 1]) * np.cos((2 * k - 1) * np.pi * y) / (2 * np.pi ** 3)

        return phi
    
    def s(self) -> float:
        """
        Gives the partial sum \sum_k=1^{k_c} f(k) appearing in \gamma = 0 Hamiltonian system.

        Returns
        -------
        transit_s: float
            Value of the sum

        Example
        -------
        >>> ds = System(4, 0.5, np.pi * np.sqrt(2), 1)
        >>> print(ds.s())
        0.2345411175184381
        """
        transit_s = 0
        for k in range(1, self._kc + 1):
            transit_s += self.f(k)

        return transit_s
    
    def Hamiltonian(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the Hamiltonian of the Pedlosky system (defined at \gamma = 0) at each time for a given solution X.

        Parameters
        ----------
        X: numpy.ndarray (of shape (n_ic, len(t_eval), self._kc + 2))
            Solution of the Pedlosky system obtained by integrating the system of ODEs. n_ic can be equal to 1.

        Returns
        -------
        H: numpy.ndarray (of shape (X.shape[1],))
            Value of the Hamiltonian at each time

        Example
        -------
        >>> kc = 1
        >>> gamma = 0
        >>> a = np.pi * np.sqrt(2)
        >>> m = 1
        >>> X0 = np.array([[1, -1.5, -1]])
        >>> time_integration = np.linspace(0, 1, 6)
        >>> ds = System(kc, gamma, a, m)
        >>> solution = ds.integration_system(time_integration, X0)
        >>> print(ds.Hamiltonian(solution))
        [0.59497891 0.59497889 0.59497887 0.59497887 0.59497888 0.59497888]
        """
        A_list = X[0, :, 0]
        B_list = X[0, :, 1]
        A0 = X[0, 0, 0]
        s_value = self.s()
        H = 0.5 * (B_list ** 2) - 0.5 * (1 + s_value * A0 ** 2) * (A_list ** 2) + 0.25 * s_value * (A_list ** 4)

        return H
    
    def Jacobian(self, X: np.ndarray) -> np.ndarray:
        return Jacobian_function(X, self._kc, self._gamma, self._a, self._m)

    def infinite_equilibrium_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the two non-trivial equilibrium points in the limit of infinite cut-off kc.

        Returns
        -------
        XP: np.ndarray (of shape (kc + 2,))
            Non-trivial equilibrium point X_+ in the limit of infinite cut-off kc.

        XM: np.ndarray (of shape (kc + 2,))
            Non-trivial equilibrium point X_- in the limit of infinite cut-off kc.

        Example
        -------
        >>> ds = System(4, 0.5, np.pi * np.sqrt(2), 1)
        >>> print(ds.infinite_equilibrium_points())
        (array([1., 0.5, 5., 1.44444444, 1.16, 1.08163265]), 
         array([-1., -0.5, 5., 1.44444444, 1.16, 1.08163265]))
        """
        XP = np.zeros(self._kc + 2)
        XM = np.zeros(self._kc + 2) 

        XP[0] = 1
        XM[0] = - 1

        XP[1] = self._gamma
        XM[1] = - self._gamma

        for i in range(1, self._kc + 1):
            XP[i + 1] = self.g(i) / self.h(i)
            XM[i + 1] = XP[i + 1]

        return XP, XM    
    
    def exact_equilibrium_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the two exact non-trivial equilibrium points in the limit of finite cut-off kc.

        Returns
        -------
        XP: np.ndarray (of shape (kc + 2,))
            Exact non-trivial equilibrium point X_+ in the limit of finite cut-off kc.

        XM: np.ndarray (of shape (kc + 2,))
            Exact non-trivial equilibrium point X_- in the limit of finite cut-off kc.

        Example
        -------
        >>> ds = System(4, 0.5, np.pi * np.sqrt(2), 1)
        >>> print(ds.exact_equilibrium_points())
        (array([1.00110439, 0.50055219, 5.01104995, 1.44763665, 1.16256359, 1.08402305])
         array([-1.00110439, -0.50055219, 5.01104995, 1.44763665, 1.16256359, 1.08402305]))
        """
        XP = np.zeros(self._kc + 2)
        XM = np.zeros(self._kc + 2) 

        total = 0
        for i in range(1, self._kc + 1):
            total += self.f(i) * (1 + self.g(i) / self.h(i))

        AP = np.sqrt(1 / total)
        XP[0] = AP
        XM[0] = - AP

        XP[1] = self._gamma * XP[0]
        XM[1] = self._gamma * XM[0]

        for i in range(1, self._kc + 1):
            XP[i + 1] = (self.g(i) / self.h(i)) * XP[0] ** 2
            XM[i + 1] = (self.g(i) / self.h(i)) * XP[0] ** 2

        return XP, XM

    def integration_system(self, t_eval: np.ndarray, ic_system: np.ndarray, parallel: bool = True) -> np.ndarray:
        """
        Integrates Pedlosky ODE system given a time grid and a set of initial conditions.

        Parameters
        ----------
        t_eval: numpy.ndarray
            Times at which the solution is evaluated
        
        ic_system: numpy.ndarray (of shape (n_ic, self._kc + 2,))
            Vector of initial conditions

        parallel: bool
            Whether to integrate the trajectories in parallel or not.

        Returns
        -------
        solution: numpy.ndarray (of shape (n_ic, len(t_eval), self._kc + 2))
            The solution of the ODE system at times specified in t_eval

        Example
        -------
        >>> kc = 2
        >>> gamma = 0.1
        >>> a = np.pi * np.sqrt(2)
        >>> m = 1
        >>> X0 = np.array([[1, -1.5, -1, 1]])
        >>> time = np.linspace(0, 1, 6)
        >>> ds = System(kc, gamma, a, m)
        >>> print(ds.integration_system(time, X0))
        [[[ 1.         -1.5        -1.          1.        ]
          [ 0.69943709 -1.3419512  -0.96930452  1.00073188]
          [ 0.43236467 -1.22337767 -0.95212387  0.99206276]
          [ 0.18958256 -1.15034994 -0.94245794  0.97831487]
          [-0.03856754 -1.12392511 -0.93586718  0.96266759]
          [-0.2615075  -1.14288154 -0.92876173  0.94767029]]]
        """
        func = system_ODE.address
        p = np.array([self._kc, self._gamma, self._a, self._m])

        if parallel:
            solution = Integrator(func, t_eval, ic_system, p)
        else:
            solution = SingleThreadIntegrator(func, t_eval, ic_system, p)

        return solution
    
    def bassin_attraction(self, A_max: float, dA: float, t_max: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the basin of attraction of the Pedlosky system by integrating a grid of initial conditions of the for (A, B, -A^2, -A^2, ..., -A^2).

        Parameters
        ----------
        A_max: float
            Maximum value of the wave amplitude in the grid of initial conditions

        dA: float
            Step size in the grid of initial conditions

        t_max: float
            Maximum time of integration

        dt: float
            Time step of the integration

        Returns
        -------
        X0: numpy.ndarray (of shape (l**2, self._kc + 2))
            Grid of initial conditions

        last_solution: numpy.ndarray (of shape (l**2,))
            Value of the first variable (A) at the final time of integration for each initial condition

        Example
        -------
        >>> kc = 1
        >>> gamma = 0.5
        >>> a = np.pi * np.sqrt(2)
        >>> m = 1
        >>> ds = System(kc, gamma, a, m)
        >>> t_max = 1
        >>> dt = 0.1
        >>> A_max = 1
        >>> dA = 0.5
        >>> result = ds.bassin_attraction(A_max, dA, t_max, dt)
        >>> print(result[1])
        [-1.77347816 -1.39706841 -1.00750079 -0.60723534 -0.19894159 -1.30842842
         -0.91161406 -0.50646695 -0.09580128  0.31745561 -0.82162929 -0.41230331
         0.          0.41230331  0.82162929 -0.31745561  0.09580128  0.50646695
         0.91161406  1.30842842  0.19894159  0.60723534  1.00750079  1.39706841
         1.77347816]
        """
        A0 = np.arange(- A_max, A_max + dA / 2, dA)
        l = len(A0)

        X, Y = np.meshgrid(A0, A0, indexing='ij')
        X_flat = X.ravel()
        Y_flat = Y.ravel()

        X0 = np.empty((l ** 2 , self._kc + 2))
        X0[:, 0] = X_flat
        X0[:, 1] = Y_flat
        X0[:, 2:] = - (X_flat[:, None] ** 2)
        time_integration = np.linspace(0, t_max, int(t_max / dt) + 1)   

        solution = self.integration_system(time_integration, X0)
        last_solution = solution[:, -1, 0]

        return X0, last_solution
    
    def get_Lyapunov_spectrum(self, ic_system: np.ndarray, s: int, steps: int) -> np.ndarray:
        """
        Computes the Lyapunov spectrum of the Pedlosky system for a given set of initial condition.

        Parameters
        ----------
        ic_system: numpy.ndarray (of shape (n_ic, self._kc + 2,)) (see above)
            Vector of initial conditions

        s: float
            Step size for the integration of the variational equations

        steps: int
            Number of steps for integrating the variational equations

        Returns
        -------
        Lyapunov_result: numpy.ndarray (of shape (n_ic, self._kc + 2,))
            Lyapunov spectrum of the Pedlosky system for the given initial conditions

        Example
        -------
        >>> kc = 1
        >>> gamma = 0.5
        >>> a = np.pi * np.sqrt(2)
        >>> m = 1
        >>> ds = System(kc, gamma, a, m)
        >>> X0 = np.array([[1, -1.5, -1], [0.5, 0.5, -0.25], [-1, 1, -1], [2, -1, -4]])
        >>> s = 0.01
        >>> steps = 10000
        >>> print(ds.get_Lyapunov_spectrum(X0, s, steps))
        [[-0.0174137  -0.1055322  -0.79372121]
         [-0.03931313 -0.02656701 -0.85078703]
         [ 0.00974479 -0.05677824 -0.86963377]
         [-0.02966311 -0.03854973 -0.84845423]]
        """
        system = system_tangent
        p = np.array([self._kc + 2, self._kc, self._gamma, self._a, self._m])
        Lyapunov_result = spectrum_Lyapunov(system, p, s, steps, ic_system)

        return Lyapunov_result[0]
    
