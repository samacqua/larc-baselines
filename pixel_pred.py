import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arc import show_arc_task, load_arc_ios
from larc_dataset import LARC_Cell_Dataset
from larc_encoder import LARCEncoder

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NO_PRED_VAL = 11


class PredictCell(nn.Module):
    def __init__(self):
        super().__init__()

        #  --> 5x64
        self.encoder = LARCEncoder()

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


    def forward(self, io_grids, test_in, desc_tokens, x, y, **_):

        # run grids + desc through encoder
        t_out = self.encoder(io_grids, test_in, desc_tokens)

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
            grid = np.full((w, h), NO_PRED_VAL)
            print(preds)
            for (x, y), col in preds.items():
                grid[x][y] = col

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
    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=[1], resize=(30, 30))
    # predictor = torch.from
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
