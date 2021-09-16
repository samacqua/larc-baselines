import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
import json

PAD_VAL = 10


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


def gen_larc_tasks(larc_path, tasks_subset=None):
    """
    generator of LARC tasks
    :param larc_path: path to folder with LARC data
    :param tasks_subset: subset of tasks to include (number of task). If None, will use whole dataset.
    :yields: LARC task dictionary
    """

    tasks_subset = set(range(400)) if tasks_subset is None else tasks_subset    # if unspecified, use all tasks

    for task_num in tasks_subset:
        fpath = os.path.join(larc_path, f'{task_num}.json')

        with open(fpath, 'r') as f:
            yield {**json.load(f), 'num': task_num}


def gen_larc_descs(larc_path, min_suc=0.1, tasks_subset=None):
    """
    generate LARC descriptions
    :param larc_path: path to folder with LARC data
    :param min_suc: min fraction of successful builds to include in dataset
    :param tasks_subset: subset of tasks to include (number of task). If None, will use whole dataset.
    :yields: LARC description + info about task
    """
    for task in gen_larc_tasks(larc_path, tasks_subset=tasks_subset):

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


def gen_larc_descs_pytorch(larc_path, num_ios=3, resize=(30, 30), min_suc=0.1, tasks_subset=None, device='cpu'):
    """
    generate LARC descriptions with more constraints for pytorch format. turns 2-d arc grids to one-hot encoded tensors.
        pads ARC grids, ensures same number example IOs per task.
    :param larc_path: path to folder with LARC data
    :param min_suc: min fraction of successful builds to include in dataset
    :param tasks_subset: subset of tasks to include (number of task). If None, will use whole dataset.
    :yields: LARC description + info about task
    """

    tasks_subset = set(tasks_subset) if tasks_subset is not None else None  # for O(1) checks

    # pad all grids with 10s to make same size
    for i, larc_pred_task in enumerate(gen_larc_descs(larc_path, min_suc=min_suc, tasks_subset=tasks_subset)):

        new_task = larc_pred_task.copy()

        # pad IOs
        new_ios = []
        io_exs = larc_pred_task['io_grids'][:num_ios] if num_ios is not None else larc_pred_task['io_grids']
        for io_in, io_out in io_exs:
            io_in_padded = pad_grid(io_in, new_shape=resize) if resize is not None else np.array(io_in)
            io_out_padded = pad_grid(io_out, new_shape=resize) if resize is not None else np.array(io_out)

            # make grid one-hot tensor
            new_ios.append((arc2torch(io_in_padded, device=device),
                            arc2torch(io_out_padded, device=device)))

        # ensure same number IO examples per task (give 1x1 placeholder task)
        if num_ios is not None and len(new_ios) < num_ios:
            new_ios += [(arc2torch(np.full((30, 30), PAD_VAL), device=device),
                         arc2torch(np.full((30, 30), PAD_VAL), device=device)) for _ in range(num_ios - len(new_ios))]
        new_task['io_grids'] = new_ios

        # pad test IO
        new_task['output_size'] = len(larc_pred_task['test'][1]), len(larc_pred_task['test'][1][0])
        test_in, test_out = larc_pred_task['test']
        new_task['test'] = (arc2torch(np.array(test_in), device=device), torch.tensor(test_out, device=device)) if resize is None \
                      else (arc2torch(pad_grid(test_in, resize), device=device), torch.tensor(pad_grid(test_out, resize), device=device))

        yield new_task


class LARCSingleCellDataset(Dataset):
    """dataset for predicting each cell color in LARC dataset."""

    def __init__(self, larc_path, resize=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf')):
        """
        Params:
            larc_path: path to folder with LARC data in it
            resize: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []
        for i, larc_pred_task in enumerate(gen_larc_descs_pytorch(larc_path, num_ios=num_ios, resize=resize,
                                                                  min_suc=0.1, tasks_subset=tasks_subset)):

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            num_cats, w, h = larc_pred_task['test'][0].shape
            for x in range(w):
                for y in range(h):
                    # 1-hot x and y
                    max_x, max_y = resize
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

    def __init__(self, larc_path, resize=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf'), device='cpu'):
        """
        Params:
            larc_path: path to folder with LARC data in it
            resize: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []

        for i, larc_pred_task in enumerate(gen_larc_descs_pytorch(larc_path, num_ios=num_ios, resize=resize,
                                                                  min_suc=0.1, tasks_subset=tasks_subset, device=device)):

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            self.tasks.append(larc_pred_task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def larc_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

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
    desc_tokens = {k: torch.tensor(v) for k, v in tokenizer.batch_encode_plus(descs, padding=True).items()}

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
    print(desc_tokens_dummy_loc['input_ids'].shape)

    return {'io_grids': io_grids, 'test_in': test_in, 'test_out': test_out, 'desc_tokens': desc_tokens_dummy_loc,
            'metadata': metadata}

if __name__ == '__main__':
    ds = LARCDataset('larc', max_tasks=6)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, collate_fn=larc_collate_dummy_language)
    for t in dl:
        print(t.keys())
