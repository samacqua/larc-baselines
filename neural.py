import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ARC_Grid_Dataset(Dataset):
    """ARC dataset."""

    def __init__(self, tasks_json_path):
        """
        :param tasks_json_path: path to folder with task jsons in it
        """

        # convert all ARC tasks to tensors
        self.grids = []
        for task in self.gen_arc_tasks(tasks_json_path):
            for io_in, io_out in task['io']:
                self.grids.append(torch.tensor(io_in))
                self.grids.append(torch.tensor(io_out))

            self.grids.append(torch.tensor(task['test_in']))
            self.grids.append(torch.tensor(task['test_out']))

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.grids[idx]

    def gen_arc_tasks(self, task_json_path):
        """
        generator for arc tasks for input to cnn autoencoder
        :param task_json_path: path to folder with LARC tasks
        :yields: [(input1, output1), (input2, output2)...], test_input, test output
        """
        for fname in os.listdir(task_json_path):
            with open(os.path.join(task_json_path, fname), 'r') as f:
                task = json.load(f)
                io_exs = []

                # get examples IOs
                for t in task['train']:
                    io_exs.append((t['input'], t['output']))

                # get test IO
                test_in = task['test'][0]['input']
                test_out = task['test'][0]['output']

                yield {'io': io_exs, 'test_in': test_in, 'test_out': test_out}

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x

def gen_larc_tasks(task_json_path, min_perc=0.1):
    """
    generator for tasks for input to NN
    :param task_json_path: path to folder with LARC tasks
    :yields: [(input1, output1), (input2, output2)...], test_input, test output, NL description
    """
    for fname in os.listdir(task_json_path):
        with open(os.path.join(task_json_path, fname), 'r') as f:
            task = json.load(f)
            io_exs = []

            # get examples IOs
            for t in task['train']:
                io_exs.append((t['input'], t['output']))

            # get test IO
            io_test = (task['test'][0]['input'], task['test'][0]['output'])

            # yield for each description
            for desc in task['descriptions'].values():
                suc, tot = 0, 0
                for build in desc['builds'].values():
                    suc += 1 if build['success'] else 0
                    tot += 1
                if tot > 0 and suc / tot >= min_perc:   # tot > 0 to avoid / 0
                    yield io_exs, *io_test, desc['do_description']

def augment_arc_task(task):
    """
    generator to augment tasks
    :param task: [(input1, output1), (input2, output2)...], test_input, test output, NL description to augment
    :yields: each augmented task
    """
    yield task

def gen_arc_pred_tasks(tasks_json_path):
    """
    generate prediction tasks for arc tasks, predicting the color for each x, y, cell in the test grid
    :param tasks_json_path: path to folder with LARC tasks
    :yields: [(input1, output1), (input2, output2)...], test_input, test output, NL description, x, y, cell_color
    """

    for base_task in gen_larc_tasks(tasks_json_path):
        for task in augment_arc_task(base_task):
            train, test_in, test_out, nl = task

            # create prediction task for each coordinate
            for y, row in enumerate(test_out):
                for x, cell_color in enumerate(row):
                    yield train, test_in, nl, x, y, cell_color

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

if __name__ == '__main__':
    tasks_dir = 'tasks_json'
    device = get_device()

    # define model + get gpu if available
    autoencoder = ConvAutoencoder()
    autoencoder.to(device)

    # define hyperparameters
    ae_epochs = 1
    ae_criterion = nn.BCELoss()
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # define grid dataset loader
    arc_grid_dataset = ARC_Grid_Dataset(tasks_dir)
    arc_grid_dataloader = DataLoader(arc_grid_dataset, batch_size=1, collate_fn=lambda x: x)

    # train CNN autoencoder
    for epoch in range(ae_epochs):

        train_loss = 0.0
        for i, batch in enumerate(arc_grid_dataloader):
            print(batch)
            # train, test_in, test_out = task
            # autoencode_grids = [grid for ex_io in train for grid in ex_io] + test_in    # get flat list of grids to feed to autoencoder
            # ae_grids_tens = torch.from_numpy(np.array(autoencode_grids))
            #
            # ae_grids_tens = ae_grids_tens.to(device)
            # ae_optimizer.zero_grad()
            outputs = autoencoder()
            # loss = ae_criterion(outputs, ae_grids_tens)
            # loss.backward()
            # ae_optimizer.step()
            # train_loss += loss.item() * ae_grids_tens.size(0)

        train_loss = train_loss / (i + 1)
        print(f'epoch: {epoch} \tloss: {round(train_loss, 5)}')

    # # train output prediction
    # for epoch in range(epochs):
    #
    #     train_loss = 0.0
    #     for i, task in enumerate(gen_arc_pred_tasks(tasks_dir)):
    #         train, test_in, nl, x, y, cell_color = task
    #         images = images.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, images)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item() * images.size(0)
    #
    #     train_loss = train_loss / len(train_loader)
    #     print(f'Epoch: {epoch} \tTraining Loss: {round(train_loss, 5)}')

    # # print dataset length
    # print(len(list(gen_arc_pred_tasks(tasks_dir))))
