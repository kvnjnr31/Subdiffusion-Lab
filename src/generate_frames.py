
import numpy as np
from plot_velocity_field import plot_velocity_frame

# Load saved simulation data
velocity_data = np.load("velocity_field.npy")      # Shape: (T, nx, ny, nz, 3)
bacterium_path = np.load("bacterium_path.npy")     # Shape: (T, 3)

# Loop through each timestep and save PNG frames
for t in range(len(velocity_data)):
    plot_velocity_frame(velocity_data[t], bacterium_path[t], t)
