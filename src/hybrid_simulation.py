import numpy as np
from lattice_boltzmann_3D import initialize_lbm, collision, streaming, macroscopic

class TIGERBacterium:
    def __init__(self, center=np.array([50, 10, 10]), Rb=1, kE=1.3, k=1.3, w=-100, h=0.77):
        self.Rb = Rb
        self.kE = kE
        self.k = k
        self.w = w
        self.h = h
        self.x = np.linspace(0, 7, 25)
        self.Ex = 1 - np.exp(-(kE**2) * self.x**2)
        self.center = np.array(center, dtype=float)
        self.theta = np.pi / 3
        self.theta_step = -np.pi / 24
        self.build_head_arc()

    def build_head_arc(self):
        head_coord_x0 = np.linspace(-2*(3*self.Rb), 0, 10)
        hy = np.zeros_like(head_coord_x0)
        hz = self.Rb * np.sqrt(1 - ((head_coord_x0 + (3 * self.Rb)) / (3 * self.Rb))**2)
        head_theta0 = np.vstack((hy, hz))
        head_coord_yz = head_theta0.copy()
        R_theta = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        for _ in range(6):
            head_theta0 = R_theta @ head_theta0
            head_coord_yz = np.hstack((head_coord_yz, head_theta0))
        self.head_coord_x = np.tile(head_coord_x0, 7)
        self.head_coord_yz = head_coord_yz
        self.R_theta_time = np.array([[np.cos(self.theta_step), -np.sin(self.theta_step)],
                                      [np.sin(self.theta_step), np.cos(self.theta_step)]])

    def update_geometry(self, t):
        self.head_coord_yz = self.R_theta_time @ self.head_coord_yz
        self.tail_y = self.h * self.Ex * np.sin(self.k * self.x - self.w * t)
        self.tail_z = self.h * self.Ex * np.cos(self.k * self.x - self.w * t)
        self.tail_x = self.x
        return self.tail_y, self.tail_z

    def advect(self, u_field, dt):
        x_idx = tuple(self.center.astype(int))
        fluid_velocity = u_field[x_idx]
        self.center += fluid_velocity * dt


def project_to_lbm(f, rho, c, w, pos, swimmer_u):
    x, y, z = np.round(pos).astype(int)
    nx, ny, nz, _ = f.shape
    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
        for i in range(len(w)):
            cu = np.dot(c[i], swimmer_u)
            feq_injection = w[i] * rho[x, y, z] * 3 * cu
            f[x, y, z, i] += feq_injection


def simulate_hybrid_lbm_tiger(nx=100, ny=30, nz=30, steps=600, tau=0.8, dt=0.05):
    f, rho, u, c, w = initialize_lbm(nx, ny, nz, tau)
    bacterium = TIGERBacterium(center=np.array([50, 15, 15]))

    velocity_frames = []
    bacterium_path = []

    for step in range(steps):

        rho, u = macroscopic(f, c)
        if step % 50 == 0:
            print(f"[{step}] Max velocity after macroscopic: {np.max(np.linalg.norm(u, axis=-1)):.3e}")

        t = step * dt
        _ = bacterium.update_geometry(t)

        # Inject swimmer momentum into fluid
        swimmer_momentum = np.array([0.1, 0.0, 0.0])
        for xi, yi, zi in zip(bacterium.tail_x, bacterium.tail_y, bacterium.tail_z):
            pos = np.round([xi, yi, zi]).astype(int)
            pos = np.clip(pos, [0, 0, 0], np.array(u.shape[:3]) - 1)
            project_to_lbm(f, rho, c, w, pos, swimmer_momentum * 0.05)

        f = collision(f, rho, u, c, w, tau)
        f = streaming(f, c)

        bacterium.advect(u, dt)
        velocity_frames.append(u.copy())
        bacterium_path.append(bacterium.center.copy())

        if step % 10 == 0:
            print(f"[{step}] Max velocity in domain: {np.max(np.linalg.norm(u, axis=-1)):.3e}")
            print(f"Step {step}, Bacterium at {bacterium.center}")

    np.save('velocity_field.npy', np.array(velocity_frames))
    np.save('bacterium_path.npy', np.array(bacterium_path))

if __name__ == "__main__":
    simulate_hybrid_lbm_tiger()
