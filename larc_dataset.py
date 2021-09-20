import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
import json
from baby_larc import Identity, RecolorGrid, RecolorAllBlocks, RecolorSomeBlocks
from random import choices

PAD_VAL = 10

# =======
# helpers
# =======

def onehot_initialization(a, num_cats):
    """assumes a is np array. https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy"""
    out = np.zeros((a.size, num_cats), dtype=np.uint8)  # initialize correct size 3-d tensor of 0s
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (num_cats,)
    return out


def arc2torch(grid, num_cats=11, device='cpu'):
    """assumes grid is numpy array. convert 2-d grid of original arc format to 3-d one-hot encoded tensor"""
    grid = onehot_initialization(grid, num_cats)
    grid = np.rollaxis(grid, 2)
    return torch.from_numpy(grid).float().to(device)


def pad_grid(grid, new_shape, pad_val=PAD_VAL):
    """pad an arc grid (list)"""
    grid = np.array(grid)
    grid_padded = np.full(new_shape, pad_val)
    grid_padded[:grid.shape[0], :grid.shape[1]] = grid
    return grid_padded


# =======================
# generate / format tasks
# =======================

def gen_baby_larc_tasks(task_kinds=(Identity, RecolorGrid, RecolorAllBlocks, RecolorSomeBlocks), max_tasks=100,
                        min_grid_size=(3, 3), max_grid_size=(10, 10), seed=None, num_ios=3):
    """
    generator of easy LARC tasks
    :param task_kinds: the kinds of tasks to generate tasks from
    :param max_tasks: the total number of tasks to yield
    :param min_grid_size: the minimum grid size to create
    :param max_grid_size: the maximum grid size to create
    :yields: LARC task dictionary
    """

    for i in range(max_tasks):
        task_kind = task_kinds[i % len(task_kinds)]     # loop through the list of task types

        # create the task, passing correct parameters based on task type
        if task_kind == Identity:
            task = Identity(num_ios=num_ios, min_grid_size=min_grid_size, max_grid_size=max_grid_size, seed=seed)
        else:
            from_color, to_color = choices(range(10), k=2)  # randomly choose colors
            task = task_kind(num_ios=num_ios, from_color=from_color, to_color=to_color, min_grid_size=min_grid_size,
                             max_grid_size=max_grid_size, seed=seed)

        task.show()

        # put into same format as actual LARC description
        yield {'io_grids': task.ios, 'test': task.test, 'desc': task.desc, 'desc_id': 'auto', 'num': i, 'name': f'{type(task).__name__} {i}'}


def gen_larc_desc_tasks(larc_path, min_suc=0.1, tasks_subset=None):
    """
    generate LARC descriptions
    :param larc_path: path to folder with LARC data
    :param min_suc: min fraction of successful builds to include in dataset
    :param tasks_subset: subset of tasks to include (number of task). If None, will use whole dataset.
    :yields: LARC description + info about task
    """

    tasks_subset = set(range(400)) if tasks_subset is None else tasks_subset    # if unspecified, use all tasks
    for task_num in tasks_subset:
        fpath = os.path.join(larc_path, f'{task_num}.json')
        with open(fpath, 'r') as f:
            task = {**json.load(f), 'num': task_num}

            # get examples IOs + test IO
            io_exs = [(t['input'], t['output']) for t in task['train']]
            io_test = (task['test'][0]['input'], task['test'][0]['output'])

            # yield for each description that meets criterion
            for desc_id, desc_obj in task['descriptions'].items():
                num_suc = len([build for build in desc_obj['builds'].values() if build['success']])
                num_tot = len(desc_obj['builds'].values())

                if num_tot > 0 and num_suc / num_tot >= min_suc:  # tot > 0 to avoid / 0
                    yield {'io_grids': io_exs, 'test': io_test, 'desc': desc_obj, 'desc_id': desc_id,
                           'num': task['num'], 'name': task['name']}


def gen_larc_tasks_pytorch(larc_gen_func, num_ios=3, max_size=(30, 30), device='cpu'):
    """
    generate LARC tasks for each description with more constraints for pytorch format. turns 2-d arc grids to one-hot
        encoded tensors. pads ARC grids, ensures same number example IOs per task.
    :param larc_gen_func: generating function to generate LARC tasks (ex: gen_larc_tasks or gen_baby_larc_tasks)
    :param num_ios: the number of input outputs to include when encoding task
    :param max_size: the max size of grids to include in the dataset, rest of grids padded to this size
    :yields: LARC description + info about task
    """

    # pad all grids with 10s to make same size, remove tasks with grids too big
    for i, larc_pred_task in enumerate(larc_gen_func()):

        # ignore if grids too big
        grids = larc_pred_task['io_grids'] + [larc_pred_task['test']]
        max_grid_w = max([max(len(io_in), len(io_out)) for io_in, io_out in grids])
        max_grid_h = max([max(len(io_in[0]), len(io_out[0])) for io_in, io_out in grids])
        if max_grid_w > max_size[0] or max_grid_h > max_size[1]:
            continue

        new_task = larc_pred_task.copy()

        # pad IOs
        new_ios = []
        io_exs = larc_pred_task['io_grids'][:num_ios] if num_ios is not None else larc_pred_task['io_grids']
        for io_in, io_out in io_exs:
            io_in_padded = pad_grid(io_in, new_shape=max_size) if max_size is not None else np.array(io_in)
            io_out_padded = pad_grid(io_out, new_shape=max_size) if max_size is not None else np.array(io_out)

            # make grid one-hot tensor
            new_ios.append((arc2torch(io_in_padded, device=device),
                            arc2torch(io_out_padded, device=device)))

        # ensure same number IO examples per task (give 1x1 placeholder task)
        if num_ios is not None and len(new_ios) < num_ios:
            new_ios += [(arc2torch(np.full(max_size, PAD_VAL), device=device),
                         arc2torch(np.full(max_size, PAD_VAL), device=device)) for _ in range(num_ios - len(new_ios))]
        new_task['io_grids'] = new_ios

        # pad test IO
        new_task['output_size'] = len(larc_pred_task['test'][1]), len(larc_pred_task['test'][1][0])
        test_in, test_out = larc_pred_task['test']
        new_task['test'] = (arc2torch(np.array(test_in), device=device), torch.tensor(test_out, device=device)) if max_size is None \
                      else (arc2torch(pad_grid(test_in, max_size), device=device), torch.tensor(pad_grid(test_out, max_size), device=device))

        yield new_task


# =======================
# create pytorch datasets
# =======================

class LARCSingleCellDataset(Dataset):
    """dataset for predicting each cell color in LARC dataset."""

    def __init__(self, larc_path, max_size=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf')):
        """
        Params:
            larc_path: path to folder with LARC data in it
            max_size: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []
        for i, larc_pred_task in enumerate(gen_larc_tasks_pytorch(larc_path, num_ios=num_ios, max_size=max_size,
                                                                  min_suc=0.1, tasks_subset=tasks_subset)):

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            num_cats, w, h = larc_pred_task['test'][0].shape
            for x in range(w):
                for y in range(h):
                    # 1-hot x and y
                    max_x, max_y = max_size
                    larc_pred_task['x'] = torch.zeros(max_x)
                    larc_pred_task['x'][x] = 1
                    larc_pred_task['y'] = torch.zeros(max_y)
                    larc_pred_task['y'][y] = 1

                    larc_pred_task['col'] = larc_pred_task['test'][1][x][y]

                    self.tasks.append(larc_pred_task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]


class LARCDataset(Dataset):
    """dataset for predicting test output grid in LARC dataset."""

    def __init__(self, larc_path, max_size=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf'), device='cpu'):
        """
        Params:
            larc_path: path to folder with LARC data in it
            max_size: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []

        gen_func = lambda: gen_larc_desc_tasks(larc_path, min_suc=0.1, tasks_subset=tasks_subset)
        for i, larc_pred_task in enumerate(gen_larc_tasks_pytorch(gen_func, num_ios=num_ios, max_size=max_size,
                                                                  device=device)):
            # only load max_tasks tasks
            if i >= max_tasks:
                break

            self.tasks.append(larc_pred_task)

        if len(self.tasks) == 0:
            print('Warning: No tasks that meet the criteria. Dataset is empty.')

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]


class BabyLARCDataset(Dataset):
    """dataset for predicting test output grid in LARC dataset."""

    def __init__(self, num_ios=3, task_kinds=(Identity, RecolorGrid, RecolorAllBlocks, RecolorSomeBlocks),
                 max_tasks=float('inf'), device='cpu', min_grid_size=(3, 3), max_grid_size=(10, 10), seed=None):
        """
        Params:
            max_size: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []

        gen_func = lambda: gen_baby_larc_tasks(task_kinds=task_kinds, max_tasks=max_tasks, min_grid_size=min_grid_size,
                                               max_grid_size=max_grid_size, seed=seed, num_ios=num_ios)

        self.tasks = [t for t in gen_larc_tasks_pytorch(gen_func, num_ios=num_ios, max_size=max_grid_size, device=device)]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]


# ========================
# collate through datasets
# ========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def larc_collate(batch, num_ios=3, device='cpu'):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    # get all data together
    io_grids, test_in, test_out, descs, metadata = [([], []) for _ in range(num_ios)], [], [], [], []
    for b in batch:

        # add IO input and output
        for i in range(num_ios):
            io_grids[i][0].append(b['io_grids'][i][0])
            io_grids[i][1].append(b['io_grids'][i][1])

        # add test IO
        test_in.append(b['test'][0])
        test_out.append(b['test'][1])

        # add desc tokens
        descs.append(b['desc']['do_description'])

        # store metadata
        metadata.append({'num': b['num'], 'desc_id': b['desc_id']})

    # convert to tensors
    io_grids = [(torch.stack(io_grids[i][0]), torch.stack(io_grids[i][1])) for i in range(num_ios)]
    test_in = torch.stack(test_in)
    test_out = torch.stack(test_out)
    desc_tokens = {k: torch.tensor(v, device=device) for k, v in tokenizer.batch_encode_plus(descs, padding=True).items()}  # make sure to put tensors to device

    return {'io_grids': io_grids, 'test_in': test_in, 'test_out': test_out, 'desc_tokens': desc_tokens, 'metadata': metadata}


desc_tokens_dummy = {k: torch.tensor(v) for k, v in tokenizer.encode_plus(['dummy'], padding=True).items()}
def larc_collate_dummy_language(batch):
    r"""Puts each data field into a tensor with outer dimension batch size, does not actually encode data (for speed/reference)"""
    # get all data together
    io_grids, test_in, test_out, descs, metadata = [([], []), ([], []), ([], [])], [], [], [], []
    for b in batch:

        # add IO input and output
        for i in range(3):
            io_grids[i][0].append(b['io_grids'][i][0])
            io_grids[i][1].append(b['io_grids'][i][1])

        # add test IO
        test_in.append(b['test'][0])
        test_out.append(b['test'][1])

        # add desc tokens
        descs.append(b['desc']['do_description'])

        # store metadata
        metadata.append({'num': b['num'], 'desc_id': b['desc_id']})

    # convert to tensors
    io_grids = [(torch.stack(io_grids[i][0]), torch.stack(io_grids[i][1])) for i in range(3)]
    test_in = torch.stack(test_in)
    test_out = torch.stack(test_out)
    desc_tokens_dummy_loc = {k: torch.stack([v]*len(batch)) for k, v in desc_tokens_dummy.items()}

    return {'io_grids': io_grids, 'test_in': test_in, 'test_out': test_out, 'desc_tokens': desc_tokens_dummy_loc,
            'metadata': metadata}


if __name__ == '__main__':
    larc_ds = LARCDataset('larc', max_size=(10, 10), max_tasks=20)
    print(len(larc_ds))

    baby_larc_ds = BabyLARCDataset(max_tasks=20)
    print(len(baby_larc_ds))

    from torch.utils.data import DataLoader

    bbl_dl = DataLoader(baby_larc_ds, collate_fn=larc_collate)

    for data in bbl_dl:
        print(data['test_out'])



    # for i in gen_larc_desc_tasks(larc_path='larc'):
    #     print(i.keys())
    #     print(i['desc'].keys())
    #     break
    # print("="*10)
    # for i in gen_baby_larc_tasks():
    #     print(i.keys())
    #     print(i['desc'].keys())
    #     break
