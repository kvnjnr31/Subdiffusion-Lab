
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import image

# Directory with PNG frames
frame_dir = "frames"
output_path = "fluid_bacterium_movie.mp4"
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

# Load one frame to get dimensions
sample_img = image.imread(os.path.join(frame_dir, frame_files[0]))
fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(sample_img)
ax.axis("off")

def update(i):
    img.set_data(image.imread(os.path.join(frame_dir, frame_files[i])))
    return [img]

ani = animation.FuncAnimation(fig, update, frames=len(frame_files), interval=100)
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=10, metadata=dict(artist="AILV"), bitrate=1800)
ani.save(output_path, writer=writer, dpi=200)

plt.close()
print(f"🎬 Movie saved as {output_path}")
