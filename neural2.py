import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from arc import show_arc_task, show_arc_grid, load_arc_ios

import pickle
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def onehot_initialization(a, num_cats):
    """https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy"""
    out = np.zeros((a.size, num_cats), dtype=np.uint8)  # initialize correct size 3-d tensor of 0s
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (num_cats,)
    return out


def arc2torch(grid, num_cats=11):
    """convert 2-d grid of original arc format to 3-d one-hot encoded tensor"""
    grid = onehot_initialization(grid, num_cats)
    grid = np.rollaxis(grid, 2)
    return torch.from_numpy(grid).float()


class LARC_Cell_Dataset(Dataset):
    """dataset for predicting each cell color in LARC dataset."""

    def __init__(self, tasks_json_path, max_size=(30,30), tasks_subset=None, max_tasks=float('inf')):
        """
        Params:
            tasks_json_path: path to folder with task jsons in it
            max_size: biggest grid size, will pad zeros to match this size so input to NN is same
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []
        tasks_subset = set(tasks_subset) if tasks_subset is not None else set()     # for O(1) checks
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # pad all grids with 0s to make same size
        for i, larc_pred_task in enumerate(self.gen_larc_pred_tasks(tasks_json_path, tasks_subset=tasks_subset)):    # {'io_grids': [(input1, output1), (input2, output2)...], 'test_in': test_input, 'desc': NL description, 'pos': (x, y), 'col': cell_color}

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            new_task = larc_pred_task.copy()

            # pad IOs
            new_ios = []
            num_ios = 3
            for io_in, io_out in larc_pred_task['io_grids'][:num_ios]:
                io_in = np.array(io_in)
                io_in_padded = np.full(max_size, 10)
                io_in_padded[:io_in.shape[0],:io_in.shape[1]] = io_in

                io_out = np.array(io_out)
                io_out_padded = np.full(max_size, 10)
                io_out_padded[:io_out.shape[0],:io_out.shape[1]] = io_out

                # make grid one-hot
                new_ios.append((arc2torch(io_in_padded),
                                arc2torch(io_out_padded)))

            # ensure same number IO examples per task
            if len(new_ios) < 3:
                new_ios += [(arc2torch(np.full(max_size, 10)),
                             arc2torch(np.full(max_size, 10))) for _ in range(num_ios - len(new_ios))]
            new_task['io_grids'] = new_ios

            # pad test input
            test_in = np.array(larc_pred_task['test_in'])
            test_in_padded = np.full(max_size, 10)
            test_in_padded[:test_in.shape[0], :test_in.shape[1]] = test_in
            new_task['test_in'] = arc2torch(test_in_padded)

            # tokenize description
            new_task['desc_tokens'] = {k: torch.tensor(v) for k, v in tokenizer.encode_plus(larc_pred_task['desc']).items()}

            # 1-hot x and y
            new_task['x'] = torch.zeros(max_size[0])
            new_task['x'][larc_pred_task['x']] = 1
            new_task['y'] = torch.zeros(max_size[1])
            new_task['y'][larc_pred_task['y']] = 1

            self.tasks.append(new_task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]

    def augment_larc_task(self, task):
        """
        generator to augment tasks
        :param task: {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
        :yields: each augmented task
        """
        yield task

    def gen_larc_pred_tasks(self, tasks_json_path, tasks_subset):
        """
        generate prediction tasks for larc tasks, predicting the color for each x, y, cell in the test grid
        :param tasks_json_path: path to folder with LARC tasks
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test_in': test_input, 'desc': NL description, 'x': x, 'y': y, 'col': cell_color}
        """

        for base_task in self.gen_larc_tasks(tasks_json_path, tasks_subset=tasks_subset):  # {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
            for task in self.augment_larc_task(base_task):
                test_in, test_out = task['test']

                # create prediction task for each coordinate
                for y, row in enumerate(test_out):
                    for x, cell_color in enumerate(row):
                        yield {'io_grids': task['io_grids'], 'test_in': test_in, 'desc': task['desc'], 'x': x, 'y': y,
                               'col': cell_color, 'num': task['num']}

    def gen_larc_tasks(self, task_json_path, min_perc=0.1, tasks_subset=None):
        """
        generator for tasks for input to NN
        :param task_json_path: path to folder with LARC tasks
        :min_perc minimum fraction of successful communications to include description in dataset
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
        """

        num_tasks = 0
        for fname in os.listdir(task_json_path):

            task_num = int(fname.split('.')[0])

            # if subset specified, ignore tasks not in subset
            if tasks_subset is not None and task_num not in tasks_subset:
                continue

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
                        num_tasks += 1
                        yield {'io_grids': io_exs, 'test': io_test, 'desc': desc['do_description'], 'num': task_num}


class PredictCell(nn.Module):
    def __init__(self):
        super().__init__()

        # grid encoder
        # 30x30x11 --> 256
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(),
        )

        # input vs. output embedding
        # 256 --> 128
        self.in_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.out_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # example embedding
        # 256 --> 64
        self.ex_encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # test input embedding
        # 256 --> 64
        self.test_in_embedding = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # natural language description encoding
        # nl --> 64
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.requires_grad_(False)
        self.bert_resize = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
        )

        # transformer
        # 5x64 --> 11
        encoder_layer = nn.TransformerEncoderLayer(d_model=5, nhead=5)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # x, y embedding
        # 30 --> 64
        self.x_embedding = nn.Sequential(
            nn.Linear(30, 64)
        )
        self.y_embedding = nn.Sequential(
            nn.Linear(30, 64)
        )

        # classification
        # 192 --> 11
        self.predict = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 11)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, io_grids, test_in, desc_tokens, x, y, **kwargs):

        # run grids through encoders
        transformer_input = []
        for io_in, io_out in io_grids:
            io_in = self.in_encoder(self.encoder(io_in))
            io_out = self.out_encoder(self.encoder(io_out))
            io = self.ex_encoder(torch.cat((io_in, io_out), dim=-1))

            transformer_input.append(io)

        # run test input grid through encoder
        transformer_input.append(self.test_in_embedding(self.encoder(test_in)))

        # run through BERT
        transformer_input.append(self.bert_resize(self.bert(**desc_tokens)['pooler_output']))

        # concatenate all inputs and run through transformer
        t_in = torch.transpose(torch.cat(transformer_input, dim=0), 0, 1).unsqueeze(0)
        t_out = self.transformer(t_in)

        # encode x, y
        x = self.x_embedding(x)
        y = self.y_embedding(y)

        # get predictions
        t_out = torch.max(t_out, dim=2).values
        classification_in = torch.cat((t_out, x, y), dim=1)
        out = self.predict(classification_in)

        return out


def train(model, dataset, num_epochs=5, batch_size=1, learning_rate=1e-3, print_every=20, save_every=200):
    """train pytorch classifier model"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0
        epoch_loss = 0
        for i, data in enumerate(train_loader):

            pred = model(**data)
            optimizer.zero_grad()
            loss = criterion(pred, data['col'])
            loss.backward()
            optimizer.step()

            running_loss += loss
            epoch_loss += loss

            if i % print_every == 0 and i != 0:
                print(f'epoch {epoch}.{i}:\tloss: {round(float(running_loss), 4)}')
                running_loss = 0

            if i % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'model.pt')

        epoch_losses.append(epoch_loss.item())
        print(f'epoch {epoch} loss: {round(epoch_loss.item(), 2)}')
        plt.plot(epoch_losses)
        plt.savefig('train.png')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model.pt')


def test(model, dataset, save=True):
    """test pytorch classifier model"""
    model.eval()
    test_loader = DataLoader(dataset)

    task_preds = {}

    with torch.no_grad():
        for data in tqdm(test_loader):
            preds = model(**data)
            best = torch.argmax(preds).item()

            # store the predicted color so can reconstruct total prediction
            x, y = torch.argmax(data['x']).item(), torch.argmax(data['y']).item()
            task_preds.setdefault(data['num'].item(), {})[(x, y)] = best

    # save inference results
    with open('temp.pkl', 'wb') as f:
        pickle.dump(task_preds, f)
    # with open('temp.pkl', 'rb') as f:
    #     task_preds = pickle.load(f)

    if save:
        os.makedirs('output', exist_ok=True)

        # reconstruct each task
        for task_num, preds in task_preds.items():

            w, h = max(preds.keys(), key=lambda x: x[0])[0] + 1, max(preds.keys(), key=lambda x: x[1])[1] + 1
            grid = np.zeros((w, h))
            for (x, y), col in preds.items():
                grid[y][x] = col

            # plot prediction vs. ground truth
            ex_ios, test_io = load_arc_ios(task_num)
            show_arc_task([(grid, test_io[0])], save_dir=f'output/pred_{task_num}.png', name=str(task_num))


if __name__ == '__main__':

    # ===================
    # test shapes correct
    # ===================

    # ios = [(torch.zeros((1, 11, 30, 30)), torch.zeros((1, 11, 30, 30))) for _ in range(3)]
    # test_in = torch.zeros((1, 11, 30, 30))
    # description = "you have to flip the square and make it blue."
    # x, y = F.one_hot(torch.tensor([0]), num_classes=30).float(), F.one_hot(torch.tensor([0]), num_classes=30).float()
    #
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # desc_enc = {k: torch.tensor([v]) for k, v in tokenizer.encode_plus(description).items()}
    #
    # predictor = PredictCell()
    # print(predictor(ios, test_in, desc_enc, x, y))

    # =======================
    # overfit on single batch
    # =======================

    predictor = PredictCell()
    tasks_dir = 'tasks_json'
    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=[1])
    larc_train_dataset.tasks = larc_train_dataset.tasks[:50]
    train(predictor, larc_train_dataset, num_epochs=50)
    test(predictor, larc_train_dataset)

    # =======================
    # create train/test split
    # =======================

    # np.random.seed(0)
    # tasks = np.arange(400)
    # np.random.shuffle(tasks)
    # train_frac = 0.8
    # train_data, test_data = tasks[:int(400 * train_frac)], tasks[int(400 * train_frac):]

    # =================================
    # define predictor + dataset, train
    # =================================

    # predictor = PredictCell()
    # tasks_dir = 'tasks_json'
    # larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=train_data, max_tasks=200)
    # train(predictor, larc_train_dataset, num_epochs=50)

    # ==========
    # test model
    # ==========

    # test_data = [1]
    # larc_test_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=test_data)
    # test(predictor, larc_test_dataset)
