from .Integrator import Integrator
import numpy as np
import numpy.linalg as la


def Gram_Schmidt(A):
    """
    Performs the Gram-Schmidt algorithm on the columns of the matrix A.

    Parameters
    ----------
    A: numpy.ndarray (of shape (n_mat, n, n))
        Collection of n_mat matrices, where the columns of each matrix represent the vectors to be orthogonalized

    Returns
    -------
    A: numpy.ndarray (of shape (n_mat, n, n))
        Matrix whose columns are orthogonalized (but not normalized) vectors

    norm_vec: numpy.ndarray (of shape (n_mat, n))
        Norms of the corresponding vectors computed during the orthogonalization process
    
    Example
    -------
    >>> import numpy as np
    >>> A = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[2, 1, 0], [2, 2, - 1], [0, 2, - 2]], [[3, 1, 1], [1, 3, - 1], [3, - 2, 3]]])
    >>> print(Gram_Schmidt(A))
    (
    array([[[ 1., -0.66666667, 0.],
            [ 1., 0.33333333, -0.5],
            [ 1., 0.33333333, 0.5]],
           [[ 2., -1., -0.33333333],
            [ 2.,  1., 0.33333333],
            [ 0.,  2. , -0.33333333]],
           [[ 3.,  1., -0.16541353],
            [ 1.,  3.,  0.13533835],
            [ 3., -2.,  0.12030075]]]), 
    array([[1.73205081, 0.81649658, 0.70710678],
           [2.82842712, 2.44948974, 0.57735027],
           [4.35889894, 3.74165739, 0.24525574]])
    )
    """
    A = A.astype(float)
    n_mat, n, _ = A.shape

    norm_vec = np.zeros((n_mat, n))
    
    for j in range(n):
        for k in range(j):
            norm_sq = norm_vec[:, j] ** 2
            projection = (np.sum(A[:, :, k] * A[:, :, j], axis = 1) / norm_sq)
            A[:, :, j] -= projection[:, np.newaxis] * A[:, :, k]

        norm_vec[:, j] = la.norm(A[:, :, j], axis = 1)
    
    return A, norm_vec

def MGS(A):
    """
    Performs the modified Gram-Schmidt algorithm on the columns of the matrix A.

    Parameters
    ----------
    A: numpy.ndarray (of shape (n_mat, n, n))
        Collection of n_mat matrices, where the columns of each matrix represent the vectors to be orthogonalized

    Returns
    -------
    A: numpy.ndarray (of shape (n_mat, n, n))
        Matrix whose columns are orthogonalized (but not normalized) vectors

    norm_vec: numpy.ndarray (of shape (n_mat, n))
        Norms of the corresponding vectors computed during the orthogonalization process
    
    Example
    -------
    >>> import numpy as np
    >>> A = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[2, 1, 0], [2, 2, - 1], [0, 2, - 2]], [[3, 1, 1], [1, 3, - 1], [3, - 2, 3]]])
    >>> print(MGS(A))
    (
    array([[[ 1., -0.66666667, 0.],
            [ 1., 0.33333333, -0.5],
            [ 1. , 0.33333333, 0.5]],
           [[ 2., -1., -0.33333333],
            [ 2.,  1., 0.33333333],
            [ 0.,  2. , -0.33333333]],
           [[ 3.,  1., -0.16541353],
            [ 1.,  3.,  0.13533835],
            [ 3., -2.,  0.12030075]]]), 
    array([[1.73205081, 0.81649658, 0.70710678],
           [2.82842712, 2.44948974, 0.57735027],
           [4.35889894, 3.74165739, 0.24525574]])
    )
    """
    A = A.astype(float)
    n_mat, n, _ = A.shape

    norm_vec = np.zeros((n_mat, n))
    
    for j in range(n):
        vec = A[:, :, j]
        norm_vec[:, j] = np.einsum("ij,ij->i", vec, vec) ** 0.5 #norm_vec[:, j] = la.norm(A[:, :, j], axis = 1) also works but is slower
        for k in range(j + 1, n):
            norm_sq = norm_vec[:, j] ** 2
            projection = (np.sum(A[:, :, k] * A[:, :, j], axis = 1) / norm_sq)
            A[:, :, k] -= projection[:, np.newaxis] * A[:, :, j]
    
    return A, norm_vec

def maximal_Lyapunov(system, p, s, n_steps, ic_system, ic_vec, convergence = False, step_convergence = 10):
    """
    Computes the maximal Lyapunov exponent of an N-dimensional dynamical system defined by the function 'system'. The function 'system' should include the (N + N^2) variational equations, representing the combined system 
    
            dX/dt = F(X)
            d\Phi/dt = DF(X) \Phi, 
            
    where:
     - X is the state vector. 
     - \Phi is the tangent map.
     - DF is the Jacobian matrix of the system.

    Parameters
    ----------
    system: numba.core.ccallback.CFunc
        Function defining the variational equations of the system

    p: numpy.ndarray
        Vector of parameters of the system. 'p[0]' is the dimension of the system

    s: float
        Step size for the integration of the variational equations

    n_steps: int
        Number of steps for integrating the variational equations

    ic_system: numpy.ndarray (of shape (n_ic, N))
        Initial conditions of the system. The 'n_ic' state vectors should be obtained after a transient integration

    ic_vec: numpy.ndarray (of shape (n_ic, N))
        Initial tangent vectors of the system. According to the Oseledec Theorem, these vectors can be randomly chosen

    convergence: bool
        If 'True', the function returns the convergence of the maximal Lyapunov exponent

    step_convergence: int (optional)
        Number of steps at which the maximal Lyapunov exponent is stored for the convergence analysis
    
    Returns
    -------
    max_Lya: numpy.ndarray (of shape (n_ic,))
        The maximal Lyapunov exponent of the dynamical system for each initial condition

    max_Lya_convergence: numpy.ndarray (of shape (n_ic, n_steps // step_convergence)), or None
        If 'convergence' is 'True', this array contains the convergence of the maximal Lyapunov exponent. Otherwise, returns 'None'
    
    Example
    -------
    >>> import numpy.random as rnd
    >>> p = np.array([3, 10, 28, 8 / 3])
    >>> s = 1
    >>> n_steps = 10 ** 3
    >>> ic_system = np.array([[5.49426896, 5.25360279, 23.86720028],
                              [1.30683122, 2.43746431, 17.90672309],
                              [-5.24888317, 1.72984277, 31.56656159],
                              [-6.26439726, -11.97872462, 9.4352913 ],
                              [6.56959495, 10.11725171, 17.73702538]]) # Points already lying on the attractor
    >>> ic_vec = rnd.rand(5, 3)
    >>> print(maximal_Lyapunov(system_Lyapunov_Lorenz, p, s, n_steps, ic_system, ic_vec, False))
    [0.90506381, 0.90760786, 0.9029999, 0.90377933, 0.9091805]
    """
    func = system.address
    dim = int(p[0])
    n_ic = ic_system.shape[0]
    param_system = p[1 :]

    ic_Lya =  np.hstack((ic_system, np.tile(np.eye(dim).flatten(), (n_ic, 1)))) # Initial condition of the variational equations. The first 'dim' coordinates are the initial state vector, and the next 'dim^2' coordinates are the tangent map initial conditions
    time_Lya = np.array([0, s / 2, s]) # This time grid structure is required by the integrator. Setting time_Lya = np.array([0, s]) will not work

    vec_norm = np.zeros((n_steps, n_ic)) # Array to store the norm of the tangent vectors as they evolve

    if convergence:
        max_Lya_convergence = np.zeros((n_ic, n_steps // step_convergence)) # Array to store the convergence of the maximal Lyapunov exponent

    for i in range(n_steps):

        solution = Integrator(func, time_Lya, ic_Lya, param_system) # Integrate the variational equations

        ic_Lya[:, : dim] = solution[:, -1, : dim] # Update the initial condition of the system at each step

        tangent_map = solution[:, - 1, dim :].reshape(n_ic, dim, dim) # Construct the tangent map as a collection of 'dim x dim' matrices
        transit_vector = tangent_map @ ic_vec[:, :, np.newaxis] # Update the tangent vectors
        transit_vector = transit_vector.reshape(n_ic, dim) # Reshape the tangent vector to a 'n_ic x dim' matrix
        vec_norm[i] = la.norm(transit_vector, axis = 1) # Compute the norm of the tangent vectors
        ic_vec = transit_vector / vec_norm[i][:, np.newaxis] # Normalize the tangent vectors for the next iteration

        if convergence and (i + 1) % step_convergence == 0:
            max_Lya_convergence[:, ((i + 1) - step_convergence) // step_convergence] = np.sum(np.log(vec_norm[: i + 1]), axis = 0) / (s * (i + 1))

    max_Lya = np.sum(np.log(vec_norm), axis = 0) / (s * n_steps) # Compute the maximal Lyapunov exponent

    if convergence:
        return max_Lya, max_Lya_convergence

    return max_Lya, None

def spectrum_Lyapunov(system, p, s, n_steps, ic_system, convergence = False, step_convergence = 10):
    """
    Computes the maximal Lyapunov exponent of an N-dimensional dynamical system defined by the function 'system'. The function 'system' should include the (N + N^2) variational equations, representing the combined system 
    
            dX/dt = F(X)
            d\Phi/dt = DF(X) \Phi, 
            
    where:
     - X is the state vector. 
     - \Phi is the tangent map.
     - DF is the Jacobian matrix of the system.

    Parameters
    ----------
    system: numba.core.ccallback.CFunc
        Function defining the variational equations of the system

    p: numpy.ndarray
        Vector of parameters of the system. 'p[0]' is the dimension of the system

    s: float
        Step size for the integration of the variational equations

    n_steps: int
        Number of steps for integrating the variational equations

    ic_system: numpy.ndarray (of shape (n_ic, N))
        Initial conditions of the system. The 'n_ic' state vectors should be obtained after a transient integration

    convergence: bool
        If 'True', the function returns the convergence of the maximal Lyapunov exponent

    step_convergence: int (optional)
        Number of steps at which the maximal Lyapunov exponent is stored for the convergence analysis
    
    Returns
    -------
    spectrum_Lya: numpy.ndarray (of shape (n_ic, N))
        The Lyapunov spectrum of the dynamical system

    max_Lya_convergence: numpy.ndarray (of shape (n_ic, N, n_steps // step_convergence)) if convergence = True. Else, None
        If 'convergence' is 'True', this array contains the convergence of the Lyapunov spectrum. Otherwise, returns 'None'
    
    Example
    -------
    >>> import numpy as np
    >>> p = np.array([3, 10, 28, 8 / 3])
    >>> s = 1
    >>> n_steps = 10 ** 3
    >>> ic_system = np.array([[5.49426896, 5.25360279, 23.86720028],
                              [1.30683122, 2.43746431, 17.90672309],
                              [-5.24888317, 1.72984277, 31.56656159],
                              [-6.26439726, -11.97872462, 9.4352913 ],
                              [6.56959495, 10.11725171, 17.73702538]]) # Points already lying on the attractor
    >>> print(spectrum_Lyapunov(system_Lyapunov_Lorenz, p, s, n_steps, ic_system, False))
    [[ 9.04711487e-01 -7.09668367e-04 -1.45706684e+01]
     [ 9.07616985e-01 -1.43498614e-03 -1.45728486e+01]
     [ 9.01289831e-01 -5.99737224e-04 -1.45673577e+01]
     [ 9.00009848e-01  4.48415682e-04 -1.45671253e+01]
     [ 9.10433933e-01  9.60310061e-05 -1.45771966e+01]]
    """
    func = system.address
    dim = int(p[0])
    n_ic = ic_system.shape[0]
    param_system = p[1 :]

    A = np.array([np.eye(dim) for _ in range(n_ic)]) # Initial tangent and orthogonal vectors
    ic_Lya = np.hstack((ic_system, A.reshape(n_ic, dim * dim))) # Initial condition of the variational equations. The first 'dim' coordinates are the initial state vector, and the next 'dim^2' coordinates are the tangent map initial conditions
    time_Lya = np.array([0, s / 2, s]) # This time grid structure is required by the integrator. Setting time_Lya = np.array([0, s]) will not work

    vec_norm = np.zeros((n_steps, n_ic, dim)) # Array to store the norm of the tangent vectors as they evolve

    if convergence:
        spectrum_Lya_convergence = np.zeros((n_ic, dim, n_steps // step_convergence)) # Array to store the convergence of the Lyapunov spectrum

    for i in range(n_steps):

        solution = Integrator(func, time_Lya, ic_Lya, param_system) # Integrate the variational equations

        tangent_map = solution[:, - 1, dim :].reshape(n_ic, dim, dim) # Construct the tangent map
        transit_vectors = tangent_map @ A # Evolve the tangent vectors one step forward

        transit_vectors, vec_norm[i] = MGS(transit_vectors) # Orthonormalize and computing the norms of the tangent vectors

        A = transit_vectors / vec_norm[i, :, :][:, np.newaxis] # Normalize the tangent vectors for the next iteration

        ic_Lya[:, : dim] = solution[:, -1, : dim] # Update the initial condition of the system at each step
        ic_Lya[:, dim :] = np.tile(np.eye(dim).flatten(), (n_ic, 1)) # Update the initial condition of the tangent map

        if convergence and (i + 1) % step_convergence == 0:
            spectrum_Lya_convergence[:, :, ((i + 1) - step_convergence) // step_convergence] = np.sum(np.log(vec_norm[: i + 1]), axis = 0) / (s * (i + 1))

    spectrum_Lya = np.sum(np.log(vec_norm), axis = 0) / (s * n_steps) # Compute the Lyapunov spectrum

    if convergence:
        return spectrum_Lya, spectrum_Lya_convergence
    
    return spectrum_Lya, None