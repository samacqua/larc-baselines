"""equivalent Google Colab notebook: https://colab.research.google.com/drive/1sFceEw0nrNjiOGiPFAb2KpnxK7kdpvDL?usp=sharing"""

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

NO_PRED_VAL = 10

def determine_kernel_size(h_in, w_in, h_out, w_out, stride=(1, 1), dilation=(1, 1), padding=(0, 0), output_padding=(0, 0)):
    """
    using the formulas:
        h_out = (h_in−1)*stride[0] - 2*padding[0] + dilation[0]*(kernel_size[0]−1) + output_padding[0]+1
        w_out = (w_in−1)*stride[1] - 2*padding[1] + dilation[1]*(kernel_size[1]−1) + output_padding[1]+1
    determine the kernel size to make all the other parameters true
    """
    kx = 1 + (h_out -(h_in - 1)*stride[0] + 2*padding[0] - output_padding[0] - 1) / dilation[0]
    ky = 1 + (w_out - (w_in - 1) * stride[1] + 2 * padding[1] - output_padding[1] - 1) / dilation[1]
    assert kx.is_integer() and ky.is_integer(), 'size is incompatible. try changing output_padding.'
    return int(kx), int(ky)

class DeconvNet(nn.Module):
    def __init__(self, max_grid_size=(30, 30)):
        super().__init__()

        # get intermediate grid sizes
        n_conv_t = 3
        grid_sizes_x = np.linspace(8, max_grid_size[0], n_conv_t+1, dtype=int)
        grid_sizes_y = np.linspace(8, max_grid_size[1], n_conv_t+1, dtype=int)

        # calculate kernel sizes to get correct grid size, make into layers
        layers = []
        for i in range(1, 4):
            kernel_size = determine_kernel_size(grid_sizes_x[i-1], grid_sizes_y[i-1], grid_sizes_x[i], grid_sizes_y[i])
            in_channels = 1 if i == 1 else 11
            layers.append(nn.ConvTranspose2d(in_channels, 11, kernel_size=kernel_size))
            layers.append(nn.ReLU())

        # Bx1x8x8 --> Bx11xWxH
        self.deconv = nn.Sequential(*layers)

    def forward(self, encoding):
        return self.deconv(encoding)


class PredictGrid(nn.Module):
    def __init__(self, max_grid_size=(30, 30)):
        super().__init__()
        self.max_grid_size = max_grid_size

        # ([(Bx11xWxH, Bx11xWxH), (Bx11xWxH, Bx11xWxH), (Bx11xWxH, Bx11xWxH)], Bx11xWxH, BxNL) --> 5x64
        self.encoder = LARCEncoder(max_grid_size=max_grid_size)

        # Bx1x64 --> Bx11xWxH
        self.decoder = DeconvNet(max_grid_size=max_grid_size)


    def forward(self, io_grids, test_in, desc_tokens, **_):

        # run grids + desc through encoder
        embedding = self.encoder(io_grids, test_in, desc_tokens)

        # although nice bc it is a nonlinearity, putting it on device is too slow
        # can't figure out how to get max while keeping result on same device
        # embedding = torch.max(embedding, dim=2)[1].view(-1, 1, 8, 8).type(torch.FloatTensor).to(DEVICE)

        embedding = torch.sum(embedding, dim=2).view(-1, 1, 8, 8)

        # run thru decoder
        pred = self.decoder(embedding)

        return pred


def train(model, train_dataset, num_epochs=5, batch_size=1, learning_rate=1e-3, print_every=20, save_every=200,
          plot_every=100, eval_dataset=None, checkpoint=None, device='cpu'):
    """train pytorch classifier model"""

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=NO_PRED_VAL)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda batch: larc_collate(batch, device=device))    # , num_workers=2, pin_memory=True

    # load previous model if provided
    starting_epoch = 0
    epoch_losses = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint['epoch_losses']

    batch_losses, validation_losses = [], []
    num_iter = 0
    for epoch in range(starting_epoch, num_epochs):
        running_loss = 0
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for i, batch_data in enumerate(train_loader):

            pred_output = model(**batch_data)
            test_output = batch_data['test_out']

            # propogate loss
            optimizer.zero_grad()
            loss = criterion(pred_output, test_output)
            loss.backward()
            optimizer.step()

            batch_losses.append((loss / pred_output.shape[0]).cpu().detach().numpy())    # mean loss
            running_loss += loss
            epoch_loss += loss

            # print progress, save model, plot training curve, plot predictions
            if num_iter % print_every == 0 and num_iter != 0:
                print(f'iter {num_iter}:\trunning loss: {round(float(running_loss), 4)}')
                running_loss = 0

            if save_every is not None and num_iter % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_losses': epoch_losses
                }, 'model.pt')

            if plot_every is not None and num_iter % plot_every == 0:
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

                if eval_dataset is not None:
                    reconstruct_preds(val_preds, f'train/{epoch}.{i}')

            num_iter += 1

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
    test_loader = DataLoader(dataset, collate_fn=lambda batch: larc_collate(batch, device=DEVICE))
    criterion = nn.CrossEntropyLoss(ignore_index=NO_PRED_VAL)
    losses = []

    task_preds = {}
    with torch.no_grad():
        for data in test_loader:
            pred_output = model(**data)
            test_output = data['test_out']

            grid = torch.argmax(pred_output, dim=1).view(*model.max_grid_size)
            losses.append((criterion(pred_output, test_output) / pred_output.shape[0]).cpu().detach().numpy())

            # save by task_num, then by desc_id
            for i in range(pred_output.shape[0]):
                task_preds.setdefault(data['metadata'][i]['num'], {})[data['metadata'][i]['desc_id']] = (grid, test_output[0].cpu().detach().numpy())

    # save inference results
    with open('temp.pkl', 'wb') as f:
        pickle.dump(task_preds, f)

    return losses, task_preds


def reconstruct_preds(predictions, save_dir):
    """reconstruct each task"""
    for task_num, pred_grids in predictions.items():

        for desc_id, (pred_grid, gt_grid) in pred_grids.items():

            h = next(i for i in range(gt_grid.shape[0]) if gt_grid[i][0] == NO_PRED_VAL)
            w = next(i for i in range(gt_grid.shape[1]) if gt_grid[0][i] == NO_PRED_VAL)
            pred_grid = pred_grid[:h, :w]  # only show correct dimensions
            gt_grid = gt_grid[:h, :w]

            # plot prediction vs. ground truth
            img_save_dir = os.path.join(save_dir, str(task_num))
            os.makedirs(img_save_dir, exist_ok=True)
            img_path = os.path.join(img_save_dir, f'{desc_id}.png')

            show_arc_task([(pred_grid, gt_grid)], save_dir=img_path, name=str(task_num), show=False)


if __name__ == '__main__':
    from larc_dataset import LARCDataset, BabyLARCDataset, larc_collate
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===================
    # test shapes correct
    # ===================

    # from transformers import BertTokenizer
    #
    # batch_size = 1
    # grid_x, grid_y = 10, 10
    # ios = [(torch.zeros((batch_size, 11, grid_x, grid_y), device=DEVICE), torch.zeros((batch_size, 11, grid_x, grid_y), device=DEVICE))
    #        for _ in range(3)]
    # test_in = torch.zeros((batch_size, 11, grid_x, grid_y), device=DEVICE)
    # descriptions = ["flip square and make it blue."] * (batch_size - 1) + [
    #     "turn it yellow."]  # to make sure can accept variable description lengths
    #
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # desc_enc = {k: torch.tensor(v, device=DEVICE) for k, v in
    #             tokenizer.batch_encode_plus(descriptions, padding=True).items()}
    #
    # predictor = PredictGrid(max_grid_size=(grid_x, grid_y)).to(DEVICE)
    # res = predictor(ios, test_in, desc_enc)
    # print(res.shape)

    # ========================
    # overfit on identity task
    # ========================

    from baby_larc import Identity

    grid_size = 10, 10
    predictor = PredictGrid(max_grid_size=grid_size).to(DEVICE)
    larc_train_dataset = BabyLARCDataset(max_tasks=1, max_grid_size=grid_size, task_kinds=(Identity,), seed=0)

    checkpoint = torch.load('model.pt')
    # checkpoint = None
    train(predictor, larc_train_dataset, num_epochs=400, checkpoint=checkpoint, eval_dataset=larc_train_dataset,
          save_every=20, batch_size=1, print_every=10, plot_every=10, device=DEVICE, learning_rate=1e-4)
    # with torch.autograd.profiler.profile() as prof:
    #     test(predictor, larc_train_dataset)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # =======================
    # overfit on tiny dataset
    # =======================

    # grid_size = 10, 10
    # predictor = PredictGrid(max_grid_size=grid_size).to(DEVICE)
    # tasks_dir = 'larc'
    # larc_train_dataset = LARCDataset(tasks_dir, tasks_subset=[9], max_size=grid_size)
    #
    # checkpoint = torch.load('model.pt')
    # # checkpoint = None
    # train(predictor, larc_train_dataset, num_epochs=100, checkpoint=checkpoint, eval_dataset=larc_train_dataset,
    #       save_every=20, batch_size=1, print_every=10, plot_every=10, device=DEVICE, learning_rate=1e-3)
    # with torch.autograd.profiler.profile() as prof:
    #     test(predictor, larc_train_dataset)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # =======================
    # create train/test split
    # =======================

    # np.random.seed(0)
    # tasks = np.arange(400)
    # np.random.shuffle(tasks)
    # train_frac = 0.8
    # train_data, test_data = tasks[:int(400 * train_frac)], tasks[int(400 * train_frac):]

    # =======================================
    # define predictor + dataset, train, test
    # =======================================

    # predictor = PredictGrid().to(DEVICE)
    # tasks_dir = 'larc'
    # larc_train_dataset = LARCDataset(tasks_dir, tasks_subset=train_data, resize=(30, 30))
    # train(predictor, larc_train_dataset, num_epochs=100, eval_dataset=larc_train_dataset,
    #       save_every=None, batch_size=32, print_every=1, plot_every=10)
    #
    # larc_test_dataset = LARCDataset(tasks_dir, tasks_subset=test_data, resize=(30, 30))
    # test(predictor, larc_test_dataset)
