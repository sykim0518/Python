import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(1)


fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(50))
ax.set_ylim(0, 1)


def update(data):
    line.set_ydata(data)
    return line,


def data_gen():
    while True:
        yield np.random.rand(50)

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
plt.show()