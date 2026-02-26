import numpy as np
from numba import njit, prange
from numbalsoda import lsoda


@njit(parallel=True)
def Integrator(func: callable, t_eval: np.ndarray, x0: np.ndarray, param: np.ndarray, rtol :float = 1.0e-8, atol: float = 1.0e-8, mxstep: float = 10 ** 8) -> np.ndarray:
    """
    Parameters
    ----------
    func: callable
        The vector function, possibly Numba-jitted, ''\bm{F}(t, \bm{X})'' to be integrated. Should have the signature func(t, x, dx, p), where 't' is the time, 'x' is the state vector, 'dx' is the derivative vector and 'p' is a parameter vector (see example below).

    t_eval: numpy.ndarray
        Times at which to evaluate solution

    x0: numpy.ndarray
        Initial condition(s)

    param: numpy.ndarray
        Parameter vector

    rtol: float, optional
        Relative tolerance. Default is 1.0e-8

    atol: float, optional
        Absolute tolerance. Default is 1.0e-8

    mxstep: float, optional
        Maximum number of steps. Default is np.inf

    For more information about rtol, atol and mxstep, see the documentation of the function scipy.integrate.solve_ivp.

    Returns
    -------
    solution: numpy.ndarray
        The solution of the system of differential equations at the times specified in t_eval
        
    Example
    -------

    >>> import Integrator
    >>> from numba import cfunc
    >>> from numbalsoda import lsoda_sig
    >>> import numpy as np
    >>> from numpy import random
    >>> @cfunc(lsoda_sig)
    >>> def SystemODE(t, x, dx, p): # The system to integrate
            dx[0] = p[0] * x[1]
            dx[1] = p[1] * x[0]
            dx[2] = - x[2]
    >>> Func = SystemODE.address # Get the memory address of the compiled function
    >>> p = np.array([2.0, 2.0]) # Parameters
    >>> ntrajecories = 5
    >>> x0 = random.uniform(0, 2, size=(ntrajecories,3))
    >>> t = np.array([0.0, 1.0, 2.0, 3.0])
    >>> result = Integrator.Integrate(Func, t, x0, p)
    >>> print(result.shape)
    (5, 4, 3)
    """
    nsamples = x0.shape[0]
    ndimension = x0.shape[1]
    ntimes = len(t_eval)

    solution = np.empty((nsamples, ntimes, ndimension)) # Array to store the solutions

    for i in prange(nsamples): # prange is the Numba way of parallelizing a loop

        transit = lsoda(func, x0[i], t_eval, param, rtol, atol, mxstep)[0]

        solution[i] = transit

    return solution