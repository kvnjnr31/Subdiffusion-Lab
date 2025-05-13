
import numpy as np

def initialize_lbm(nx, ny, nz, tau):
    """Initialize LBM parameters for D3Q19"""
    # Lattice velocities for D3Q19
    c = np.array([[ 0, 0, 0],
                  [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1],
                  [ 1, 1, 0], [-1, 1, 0], [ 1,-1, 0], [-1,-1, 0],
                  [ 1, 0, 1], [-1, 0, 1], [ 1, 0,-1], [-1, 0,-1],
                  [ 0, 1, 1], [ 0,-1, 1], [ 0, 1,-1], [ 0,-1,-1]])

    # Weights for D3Q19
    w = np.array([1/3] + [1/18]*6 + [1/36]*12)

    # Initialization
    rho = np.ones((nx, ny, nz))
    u = np.zeros((nx, ny, nz, 3))
    f = np.zeros((nx, ny, nz, 19))

    for i in range(19):
        cu = np.dot(u, c[i])
        f[:,:,:,i] = w[i] * rho * (1 + 3*cu + 9/2*cu**2 - 3/2*np.sum(u**2, axis=3))

    return f, rho, u, c, w

def collision(f, rho, u, c, w, tau):
    """BGK collision step"""
    feq = np.zeros_like(f)
    for i in range(19):
        cu = np.sum(u * c[i], axis=3)
        feq[:,:,:,i] = w[i] * rho * (1 + 3*cu + 9/2*cu**2 - 3/2*np.sum(u**2, axis=3))
    return f - (1.0 / tau) * (f - feq)

def streaming(f, c):
    """Stream populations along lattice vectors"""
    nx, ny, nz, _ = f.shape
    f_streamed = np.zeros_like(f)
    for i in range(19):
        f_streamed[:,:,:,i] = np.roll(f[:,:,:,i], shift=c[i], axis=(0,1,2))
    return f_streamed

def macroscopic(f, c):
    """Compute density and velocity from distribution function"""
    rho = np.sum(f, axis=3)
    u = np.zeros((*rho.shape, 3))
    for i in range(19):
        u += f[:,:,:,i][:,:,:,None] * c[i]
    u /= rho[:,:,:,None]
    return rho, u
