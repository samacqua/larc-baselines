import numpy as np
from arc import show_arc_task, ARC_COLORS
import random

# ===================
# grid intializations
# ===================

def make_grid(size, color=0):
    """make an ARC grid of size and color"""
    return np.full(shape=size, fill_value=color)

# ===========================
# single grid transformations
# ===========================

def place_random_blocks(grid, num_blocks=3, colors=0):
    """place num_blocks randomly sized/positioned blocks randomly"""
    w, h = grid.shape

    if isinstance(colors, int):
        colors = [colors] * num_blocks

    ws = np.arange(2, w)
    w_weights = ws[::-1] ** 2
    hs = np.arange(2, h)
    h_weights = hs[::-1] ** 2

    for i in range(num_blocks):
        xc, yc = np.random.randint(low=(0, 0), high=(h, w))    # randomly choose center

        # randomly choose width and height, making smaller sizes more likely
        block_w = random.choices(ws, w_weights)[0]
        block_h = random.choices(hs, h_weights)[0]

        # get corners of block, ensuring min is > 0
        x1, y1 = xc - block_w // 2, yc - block_h // 2
        x2, y2 = x1 + block_w, y1 + block_h
        x1, y1 = max(0, x1), max(0, y1)

        grid[y1:y2, x1:x2] = colors[i]

    return grid

def remap_color(grid, old_color, new_color):
    """change all cells of old_color to new_color"""
    return np.where(grid == old_color, new_color, grid)

# =========
# arc tasks
# =========

def gen_desc_skeleton():
    return {'action_sequence': [], 'attempt_jsons': [], 'builds': [], 'confidence': 0, 'description_time': 0,
            'do_description': '', 'grid_description': [], 'max_idle_time': 0, 'num_verification_attempts': 0,
            'see_description': '', 'succeeded_verification': True, 'timestamp': 0, 'uid': '', 'verification_time': 0}

def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

class Task:
    """parent class for ARC task."""
    def __init__(self, num_ios):
        grid_ios = []
        for _ in range(num_ios + 1):
            in_grid = self.create_grid()
            out_grid = self.transform_grid(in_grid)
            grid_ios.append((in_grid, out_grid))

        self.ios = grid_ios[:-1]
        self.test = grid_ios[-1]
        self.desc = self.create_desc()

    def create_grid(self):
        raise NotImplementedError

    def transform_grid(self, grid):
        raise NotImplementedError

    def create_desc(self):
        raise NotImplementedError

    def show(self):
        print(self.desc['do_description'])
        show_arc_task(ios=self.ios + [self.test], show=True)


class Identity(Task):
    """output == input"""
    
    def __init__(self, num_ios=3, create_fn=None, min_grid_size=(6, 6), max_grid_size=(10, 10), seed=None):
        set_seed(seed)
        if create_fn is None:
            grid_size = np.random.randint(low=min_grid_size, high=max_grid_size)
            # make grid with random objects of various colors
            create_fn = lambda: place_random_blocks(make_grid(grid_size, color=np.random.randint(10)), num_blocks=3, colors=random.choices(range(10), k=3))
        self.create_fn = create_fn
        super().__init__(num_ios)

    def create_grid(self):
        return self.create_fn()

    def transform_grid(self, grid):
        return grid

    def create_desc(self):
        desc = gen_desc_skeleton()
        desc['do_description'] = "Just return the input grid."
        return desc

class RecolorGrid(Task):
    """variable sized, uniform color grid --> different color, same sized grid"""
    def __init__(self, num_ios, from_color, to_color, min_grid_size=(6, 6), max_grid_size=(10, 10), seed=None):
        set_seed(seed)
        self.from_color = from_color
        self.to_color = to_color
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size

        super().__init__(num_ios)

    def create_grid(self):
        return make_grid(size=np.random.randint(low=self.min_grid_size, high=self.max_grid_size), color=self.from_color)

    def transform_grid(self, grid):
        return remap_color(grid, self.from_color, self.to_color)

    def create_desc(self):
        desc = gen_desc_skeleton()
        desc['do_description'] = f'Change the input from {ARC_COLORS[self.from_color]} to {ARC_COLORS[self.to_color]}'
        return desc

class RecolorAllBlocks(RecolorGrid):
    """uniform sized, grid w/ num_blocks blocks --> grid w/ each block recolored, same sized grid"""
    def __init__(self, num_ios, from_color, to_color, num_blocks=3, min_grid_size=(6, 6), max_grid_size=(10, 10), seed=None):
        set_seed(seed)
        self.from_color = from_color
        self.grid_size = np.random.randint(low=min_grid_size, high=max_grid_size)
        self.num_blocks = num_blocks
        super().__init__(num_ios, from_color, to_color)

    def create_grid(self):
        grid = make_grid(size=self.grid_size, color=0)
        return place_random_blocks(grid, num_blocks=self.num_blocks, colors=self.from_color)

    def create_desc(self):
        desc = gen_desc_skeleton()
        desc['do_description'] = f"Change all of the {ARC_COLORS[self.from_color]} in the input to {ARC_COLORS[self.to_color]}"
        return desc

class RecolorSomeBlocks(RecolorAllBlocks):
    """same as RecolorAllBlocks, but not all objects in input are the same color"""
    def __init__(self, num_ios, from_color, to_color, num_blocks=3, min_grid_size=(6, 6), max_grid_size=(10, 10), seed=None):
        set_seed(seed)
        super().__init__(num_ios, from_color, to_color, num_blocks, min_grid_size, max_grid_size)

    def create_grid(self):
        grid = make_grid(size=self.grid_size, color=0)
        block_colors = []

        # make half blocks the color to change, make the rest randomly chosen
        for _ in range(self.num_blocks):
            if np.random.choice([0, 1]) == 1:
                block_colors.append(self.from_color)
            else:
                block_colors.append(np.random.randint(10))
        return place_random_blocks(grid, num_blocks=self.num_blocks, colors=block_colors)


if __name__ == '__main__':

    identity = Identity(num_ios=2)
    identity.show()

    recolor_all = RecolorGrid(num_ios=3, from_color=1, to_color=2)
    recolor_all.show()

    recolor_all_blocks = RecolorAllBlocks(num_ios=3, from_color=5, to_color=3)
    recolor_all_blocks.show()

    recolor_blocks = RecolorSomeBlocks(num_ios=3, from_color=9, to_color=8)
    recolor_blocks.show()
