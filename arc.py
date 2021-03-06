import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import os
import torch

ARC_COLORS = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'purple', 'orange', 'teal', 'brown', 'white', 'lightgray']

def load_arc_ios(num, tasks_dir='larc'):
    """
    load ARC IO examples + test IO from LARC data
    :param num: task number
    :param tasks_dir: directory to LARC data
    :return: ([(io1_in, io1_out), (io2_in, io2_out), ...], (io_test_in, io_test_out))
    """
    with open(os.path.join(tasks_dir, f'{num}.json'), 'r') as f:
        task = json.load(f)

    io_exs = []

    # get examples IOs
    for t in task['train']:
        io_exs.append((t['input'], t['output']))

    # get test IO
    io_test = (task['test'][0]['input'], task['test'][0]['output'])
    
    return io_exs, io_test



def show_arc_task(ios, name='', save_dir=None, show=False):
    """show/save images of a collection of arc input-outputs"""
    # make subplots
    plt.clf()
    fig, ax = plt.subplots(len(ios), 2, figsize=(5, len(ios)*2))

    # show each grid
    for i, io in enumerate(ios):
        for j in range(2):
            grid = io[j]
            show_arc_grid(grid, ax[i][j]) if len(ios) > 1 else show_arc_grid(grid, ax[j])

    plt.suptitle(f'Task {name}')

    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)
    
def show_arc_grid(grid, ax=None, save_dir=None, show=False):
    """show/save images of an arc grid"""
    if ax is None:
        fig, ax = plt.subplots()

    colormap = mpl.colors.ListedColormap(ARC_COLORS)
    bounds = list(range(12))
    norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
    ax.imshow(grid, cmap=colormap, norm=norm)

    # Major ticks
    ax.set_xticks(np.arange(0, len(grid[0]), 1))
    ax.set_yticks(np.arange(0, len(grid), 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.2)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)

if __name__ == '__main__':
    ex_ios, test_io = load_arc_ios(0)
    show_arc_task(ex_ios + [test_io], show=True)

    # ex_ios = [([[1,2,3], [4,5,10]], [[1,2,3], [4,5,6]])]
    # show_arc_task(ex_ios, show=True)
