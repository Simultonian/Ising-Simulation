import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a sample heatmap data (you can replace this with your own data)
data = np.random.rand(5, 5)

# Create a sample grid array with values between 0 and 3 (you can replace this with your own grid array)
grid_array = np.array(
    [
        [0, 1, 2, 3, 0],
        [1, 2, 3, 0, 1],
        [2, 3, 0, 1, 2],
        [3, 0, 1, 2, 3],
        [0, 1, 2, 3, 0],
    ]
)

# Define a custom color map for the grid values
colors = ["red", "green", "blue", "black"]
cmap = sns.color_palette(colors)

# Create the heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(data, annot=True, cmap=cmap, cbar=False, linewidths=0.5, linecolor="gray")

# Set the grid values as tick labels
plt.xticks(np.arange(0.5, 5.5), grid_array, rotation=0)
plt.yticks(np.arange(0.5, 5.5), grid_array[::-1], rotation=0)

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Custom Color Heatmap")

# Show the plot
plt.savefig("map.png", dpi=300)
