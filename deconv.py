import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arc import show_arc_task, load_arc_ios
from larc_encoder import LARCEncoder

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

NO_PRED_VAL = 11

class DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # h_out = (h_in−1)*stride[0] - 2*padding[0] + dilation[0]*(kernel_size[0]−1) + output_padding[0]+1
        # w_out = (w_in−1)*stride[1] - 2*padding[1] + dilation[1]*(kernel_size[1]−1) + output_padding[1]+1

        # Bx1x8x8 --> Bx11x30x30
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 11, kernel_size=3),   # Bx1x8x8 --> Bx11x10x10
            nn.ReLU(),
            nn.ConvTranspose2d(11, 11, kernel_size=5, stride=2, output_padding=1),  # Bx11x10x10 --> Bx11x24x24
            nn.ReLU(),
            nn.ConvTranspose2d(11, 11, kernel_size=7),  # Bx11x24x24 --> Bx11x30x30
        )

    def forward(self, encoding):
        return self.deconv1(encoding)


class PredictGrid(nn.Module):
    def __init__(self):
        super().__init__()

        # --> 5x64
        self.encoder = LARCEncoder()

        # --> 30x30x11
        self.decoder = DeconvNet()


    def forward(self, io_grids, test_in, desc_tokens, **_):

        # run grids + desc through encoder
        embedding = self.encoder(io_grids, test_in, desc_tokens)
        embedding = torch.max(embedding, dim=2)[1].view(-1, 1, 8, 8).type(torch.FloatTensor)

        # run thru decoder
        pred = self.decoder(embedding)

        return pred


def train(model, train_dataset, num_epochs=5, batch_size=1, learning_rate=1e-3, print_every=20, save_every=200,
          plot_every=100, eval_dataset=None, checkpoint=None):
    """train pytorch classifier model"""

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=larc_collate)

    # load previous model if provided
    starting_epoch = 0
    epoch_losses = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint['epoch_losses']

    batch_losses, validation_losses = [], []
    for epoch in range(starting_epoch, num_epochs):
        running_loss = 0
        epoch_loss = 0
        for i, batch_data in enumerate(train_loader):
            pred_output = model(**batch_data)
            test_output = batch_data['test_out']

            # TODO: redo logic with multiple batches
            # # reshape so only care about grid that is of correct size
            # test_h, test_w = [i.item() for i in data['output_size']]
            # test_output = test_output[:, :test_h, :test_w]
            # pred_output = pred_output[:, :, :test_h, :test_w]

            # propogate loss
            optimizer.zero_grad()
            loss = criterion(pred_output, test_output)
            loss.backward()
            optimizer.step()

            batch_losses.append((loss / pred_output.shape[0]).detach().numpy())    # mean loss
            running_loss += loss
            epoch_loss += loss

            # print, save, plot
            if i % print_every == 0 and i != 0:
                print(f'epoch {epoch}.{i}:\trunning loss: {round(float(running_loss), 4)}')
                running_loss = 0

            if save_every is not None and i % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_losses': epoch_losses
                }, 'model.pt')

            if plot_every is not None and i % plot_every == 0:
                plt.clf()
                fig, ax = plt.subplots()
                ax.set(title='loss', xlabel='iteration num', ylabel='loss')
                ax.plot(batch_losses, color='b', label='train loss')
                if eval_dataset is not None:
                    val_losses, val_preds = test(model, eval_dataset)
                    validation_losses.append(np.mean(val_losses))
                    ax.plot([j*plot_every for j in range(len(validation_losses))], validation_losses, 'o', color='orange', label='validation loss')
                ax.legend()
                plt.savefig('train.png')

                reconstruct_preds(val_preds, f't/{epoch}.{i}')                

        # print training loss
        print(f'epoch {epoch} loss: {round(epoch_loss.item(), 2)}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses
        }, 'model.pt')


def test(model, dataset):
    """test pytorch classifier model"""
    model.eval()
    test_loader = DataLoader(dataset, collate_fn=larc_collate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    task_preds = {}
    with torch.no_grad():
        for data in tqdm(test_loader):
            pred_output = model(**data)
            test_output = data['test_out']
            grid = torch.argmax(pred_output, dim=1).view(30, 30)
            losses.append((criterion(pred_output, test_output) / pred_output.shape[0]).detach().numpy())

            # save by task_num, then by desc_id
            for i in range(pred_output.shape[0]):
                task_preds.setdefault(data['metadata'][i]['num'], {})[data['metadata'][i]['desc_id']] = grid

    # save inference results
    with open('temp.pkl', 'wb') as f:
        pickle.dump(task_preds, f)

    return losses, task_preds


def reconstruct_preds(predictions, save_dir):
    """reconstruct each task"""
    for task_num, pred_grids in predictions.items():
        ex_ios, (test_in, test_out) = load_arc_ios(task_num)
        test_h, test_w = len(test_out), len(test_out[0])

        for desc_id, pred_grid in pred_grids.items():
            pred_grid = pred_grid[:test_h, :test_w]  # only show correct dimensions

            # plot prediction vs. ground truth
            img_save_dir = os.path.join(save_dir, str(task_num))
            os.makedirs(img_save_dir, exist_ok=True)
            img_path = os.path.join(img_save_dir, f'{desc_id}.png')
            show_arc_task([(pred_grid, test_out)], save_dir=img_path, name=str(task_num), show=False)


if __name__ == '__main__':
    from larc_dataset import LARCDataset, larc_collate

    # ===================
    # test shapes correct
    # ===================

    # from transformers import BertTokenizer
    #
    # batch_size = 3
    # ios = [(torch.zeros((batch_size, 11, 30, 30)), torch.zeros((batch_size, 11, 30, 30))) for _ in range(3)]
    # test_in = torch.zeros((batch_size, 11, 30, 30))
    # descriptions = ["flip square and make it blue."] * (batch_size - 1) + ["turn it yellow."]   # to make sure can accept variable description lengths
    #
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # desc_enc = {k: torch.tensor(v) for k, v in tokenizer.batch_encode_plus(descriptions, padding=True).items()}
    #
    # predictor = PredictGrid()
    # res = predictor(ios, test_in, desc_enc)
    # print(res.shape)

    # =======================
    # overfit on tiny dataset
    # =======================

    predictor = PredictGrid()
    tasks_dir = 'larc'
    larc_train_dataset = LARCDataset(tasks_dir, tasks_subset=[1, 2], resize=(30, 30))
    # # checkpoint = torch.load('model.pt')
    checkpoint = None
    train(predictor, larc_train_dataset, num_epochs=350, checkpoint=checkpoint, eval_dataset=larc_train_dataset,
          save_every=None, batch_size=2, print_every=1, plot_every=1)
    # test(predictor, larc_train_dataset)

    # =======================
    # create train/test split
    # =======================

    # import numpy as np
    #
    # np.random.seed(0)
    # tasks = np.arange(400)
    # np.random.shuffle(tasks)
    # train_frac = 0.8
    # train_data, test_data = tasks[:int(400 * train_frac)], tasks[int(400 * train_frac):]

    # =================================
    # define predictor + dataset, train
    # =================================

    # predictor = PredictGrid()
    # tasks_dir = 'larc'
    # larc_train_dataset = LARCDataset(tasks_dir, tasks_subset=train_data, max_tasks=200)
    # train(predictor, larc_train_dataset, num_epochs=50)

    # ==========
    # test model
    # ==========

    # test_data = [1]
    # larc_test_dataset = LARCDataset(tasks_dir, tasks_subset=test_data)
    # test(predictor, larc_test_dataset)
