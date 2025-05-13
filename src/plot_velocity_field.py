import numpy as np
import matplotlib.pyplot as plt
import os


def plot_velocity_frame(u, bacterium, t, output_dir="frames", dpi=300,
                        bacterium_velocity=None, tail_points=None):
    '''
    Plot a 2x2 panel (Left, Right, Front, Back) of the 3D velocity field around the bacterium.
    Optionally adjust for swimmer frame and overlay tail geometry.

    Parameters:
    - u: velocity field (nx, ny, nz, 3)
    - bacterium: 3D position (x, y, z)
    - t: frame index
    - output_dir: directory to save PNGs
    - dpi: resolution for saved images
    - bacterium_velocity: optional 3D array for swimmer frame transform
    - tail_points: optional list of tail 3D points for overlay
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if bacterium_velocity is not None:
        u = u - np.array(bacterium_velocity)  # Transform to swimmer frame

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"TIGER Simulation – Frame {t}", fontsize=14)

    views = ["Left", "Right", "Front", "Back"]
    slice_offsets = [-5, 5, -5, 5]
    bacterium = np.array(bacterium).astype(int)
    pos_map = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (view, offset, pos) in enumerate(zip(views, slice_offsets, pos_map)):
        ax = axs[pos]

        if view in ["Front", "Back"]:
            # Plane: (y, z) at fixed x
            slice_x = np.clip(bacterium[0] + offset, 0, u.shape[0] - 1)
            uy = u[slice_x, :, :, 1]
            uz = u[slice_x, :, :, 2]
            X, Y = np.meshgrid(np.arange(uy.shape[0]), np.arange(uy.shape[1]), indexing="ij")
            speed = np.sqrt(uy**2 + uz**2)
            ax.imshow(speed.T, origin="lower", cmap="plasma", alpha=0.8)
            ax.quiver(X, Y, uy.T, uz.T, scale=30, color='white', linewidth=0.4)
            ax.set_title(f"{view} View @ x={slice_x}")

            # Overlay tail
            if tail_points is not None:
                tail = np.array(tail_points).astype(int)
                mask = tail[:, 0] == slice_x
                ax.plot(tail[mask][:, 1], tail[mask][:, 2], 'go', markersize=3)

        elif view in ["Left", "Right"]:
            # Plane: (x, z) at fixed y
            slice_y = np.clip(bacterium[1] + offset, 0, u.shape[1] - 1)
            ux = u[:, slice_y, :, 0]
            uz = u[:, slice_y, :, 2]
            X, Y = np.meshgrid(np.arange(ux.shape[0]), np.arange(ux.shape[1]), indexing="ij")
            speed = np.sqrt(ux**2 + uz**2)
            ax.imshow(speed.T, origin="lower", cmap="plasma", alpha=0.8)
            ax.quiver(X, Y, ux.T, uz.T, scale=30, color='white', linewidth=0.4)
            ax.set_title(f"{view} View @ y={slice_y}")

            # Overlay tail
            if tail_points is not None:
                tail = np.array(tail_points).astype(int)
                mask = tail[:, 1] == slice_y
                ax.plot(tail[mask][:, 0], tail[mask][:, 2], 'go', markersize=3)

        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"frame_{t:04d}.png")
    plt.savefig(save_path, dpi=dpi)
    plt.close()
