import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def draw(env, pi, U=None, data=None):

    if not env or not pi:
        raise ValueError("mdp environment and policy must be provided for gui")

    plt.figure(1)
    ax = plt.axes()
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"])
    cmap.set_bad("k")

    matrix = [[np.nan for x in range(env.cols)] for y in range(env.rows)]

    arrows = env.to_arrows(pi)


    for s in pi:
        (x, y) = s
        v = matrix[y][x] = 0
        ax.text(x, env.rows-y-1, arrows[s], va="center", ha="center")

    if U:
        for s in U:
            (x, y) = s
            v = matrix[y][x] = round(U[s], 3)
            ax.text(x, env.rows - y - 0.7, v, va="center", ha="center")
    else:
        cmap = 'viridis'
    
    matrix.reverse()
    plt.title("Grid Map")
    plt.imshow(matrix, cmap=cmap)
    plt.show()

    if data:
        plt.figure(2)
        num_of_iteration = len(data[list(data.keys())[0]])
        x = [i for i in range(num_of_iteration)]

        y_max, y_min = 0, 0
        for state in data:
            plt.plot(x, data[state], label=state)
            y_min, y_max = min(y_min, *data[state]), max(y_max, *data[state])

        plt.title("Number of iterations vs Utility Estimates")
        plt.xlabel("Number of iterations")
        plt.ylabel("Utility Estimates")
        plt.ylim(y_min, y_max)
        #plt.legend()
        plt.show()
