import matplotlib.pyplot as plt
import numpy as np

# Example scatter data
x = np.random.rand(10)
y = np.random.rand(10)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', '^', 's', 'p', '*', '+', 'x']

# Create scatter plots
fig, ax = plt.subplots()
for i in range(len(markers)):
    ax.scatter(x, y, color=colors[i], marker=markers[i], label=f'Plot {i+1}')

# Adjust layout to make room for the table
plt.subplots_adjust(left=0.2, bottom=0.2)

# Create a table with placeholder data
cell_text = [['' for _ in range(3)] for _ in range(len(markers))]
the_table = plt.table(cellText=cell_text, colLabels=['Col1', 'Col2', 'Col3'], 
                      rowLabels=[f'Plot {i+1}' for i in range(len(markers))],
                      loc='bottom', cellLoc='center', rowLoc='center')

# Adjust table and plot size
the_table.scale(1, 1.5)
ax.set_position([0.2, 0.3, 0.7, 0.6])

# Plot markers next to the row labels
for i, color in enumerate(colors):
    ax.plot([-0.15], [0.65 - 0.06 * i], color=color, marker=markers[i], markersize=10, transform=ax.transAxes)

plt.show()

diagram_name = (
    f"plots/qiskit_trotter/trotter_6_8_table.png"
)
print(f"Saving at: {diagram_name}")
plt.savefig(diagram_name, dpi=300)
