import os
import sys

from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from utils import load_data, split, show_sample_sequence, get_scenario

from copy import deepcopy


SEQUENCE_STRIDE = 10

BATCH_SIZE = 16
RNN_HIDDEN_SIZE = 256
MLP_HIDDEN_SIZE = 1024
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCH_CONVERGENCE = 8


class RNN(nn.Module):
    def __init__(self, cell, input_size, rnn_hidden_size, mlp_hidden_size,
            output_size, num_layers, mean_inputs, std_inputs, mean_targets,
            std_targets):
        super().__init__()

        self.mean_inputs = mean_inputs
        self.std_inputs = std_inputs
        self.mean_targets = mean_targets
        self.std_targets = std_targets

        if cell == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers)
        elif cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers)
        else:
            raise NotImplementedError

        self.sequential = nn.Sequential(
            nn.Linear(rnn_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size)
        )

    def forward(self, x, h0=None):
        x = (x - self.mean_inputs) / self.std_inputs
        x, hn = self.rnn(x, h0)
        x = self.sequential(x)
        x = x * self.std_targets + self.mean_targets
        return x, hn


def masked_mse_loss(preds, targets, lengths, max_length):
    num_sequences = lengths.size(0)
    output_size = targets.size(-1)

    timesteps = torch.arange(max_length).expand(num_sequences, max_length)
    masks = (timesteps < lengths.unsqueeze(1)).T.unsqueeze(-1)

    masked_difference = (preds - targets) * masks
    return (masked_difference ** 2).sum() / (masks.sum() * output_size)


def train_rnn(rnn, opt, train_inputs, train_targets, train_lengths,
        valid_inputs, valid_targets, valid_lengths, batch_size,
        num_epoch_convergence):

    lowest_loss, num_epoch_no_improvement = float('inf'), 0
    best_weights = deepcopy(rnn.state_dict())
    train_losses, valid_losses = [], []
    train_after_epoch_losses = []  # TODO: delete this

    num_train = train_targets.size(1)
    train_max_length = train_lengths.max().item()
    valid_max_length = valid_lengths.max().item()

    epoch = 0
    while num_epoch_no_improvement <= num_epoch_convergence:

        try:
            # Training
            permutation = torch.randperm(num_train)
            train_loss = 0.0

            for i in range(0, num_train, batch_size):
                indices = permutation[i:i+batch_size]
                actual_batch_size = len(indices)

                batch_inputs = train_inputs[:, indices, :]
                batch_targets = train_targets[:, indices, :]
                batch_lengths = train_lengths[indices]

                batch_preds, _ = rnn(batch_inputs)
                loss = masked_mse_loss(batch_preds, batch_targets,
                                       batch_lengths, train_max_length)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item()

            train_loss /= int(num_train / batch_size)
            train_losses.append(train_loss)

            epoch += 1

            # Training evaluation after epoch
            with torch.no_grad():
                train_preds, _ = rnn(train_inputs)
                train_after_epoch = masked_mse_loss(train_preds,
                                                    train_targets,
                                                    train_lengths,
                                                    train_max_length)
            train_after_epoch_losses.append(train_after_epoch.item())

            # Validation
            with torch.no_grad():
                valid_preds, _ = rnn(valid_inputs)
                valid_loss = masked_mse_loss(valid_preds, valid_targets,
                                             valid_lengths, valid_max_length)
            valid_losses.append(valid_loss.item())

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                num_epoch_no_improvement = 0
                best_weights = deepcopy(rnn.state_dict())
            else:
                num_epoch_no_improvement += 1

            print('Epoch {:03d}'.format(epoch))
            print('\tTrain: {:,.4f}'.format(train_loss))
            print('\tTrain after epoch: {:,.4f}'.format(train_after_epoch))
            print('\tValid: {:,.4f}'.format(valid_loss))

        except KeyboardInterrupt:
            if len(train_after_epoch_losses) < len(train_losses):
                train_after_epoch_losses.append(None)
            if len(valid_losses) < len(train_losses):
                valid_losses.append(None)
            print()
            break

    rnn.load_state_dict(best_weights)

    return {
        'train': train_losses,
        'train_after': train_after_epoch_losses,
        'valid': valid_losses
    }


def main():
    if len(sys.argv) != 3:
        print('Usage: python3 {} NAME SCENARIO'.format(sys.argv[0]))
        sys.exit(1)
    name = 'rnn-' + sys.argv[1]
    scenario = int(sys.argv[2])

    train_dataset, valid_dataset, test_dataset, train_stats = get_scenario(
        scenario,
        recurrent=True,
        sequence_stride=SEQUENCE_STRIDE)

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset
    mean_inputs, std_inputs, mean_targets, std_targets = train_stats

    print('Training size: {}'.format(train_inputs.size(1)))
    print('Validation size: {}'.format(valid_inputs.size(1)))
    print('Test size: {}'.format(test_inputs.size(1)))

    rnn = RNN(
        cell='gru',
        input_size=train_inputs.size(2),
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        output_size=train_targets.size(2),
        num_layers=NUM_LAYERS,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    opt = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

    start = perf_counter()
    stats = train_rnn(rnn, opt, train_inputs, train_targets, train_lengths,
        valid_inputs, valid_targets, valid_lengths, BATCH_SIZE,
        NUM_EPOCH_CONVERGENCE)
    end = perf_counter()

    os.makedirs('results/', exist_ok=True)
    df = pd.DataFrame.from_dict(stats)
    df.to_csv('results/{}.csv'.format(name))

    print('Total CPU time {:.2f}'.format(end - start))
    print('CPU time per epoch {:.2f}'.format((end - start) / len(df)))

    with torch.no_grad():
        test_preds, _ = rnn(test_inputs)

    mse_loss = masked_mse_loss(test_preds, test_targets, test_lengths,
                               test_lengths.max().item())
    print('Test MSE Loss: {:,.4f}'.format(mse_loss.item()))

    os.makedirs('weights/', exist_ok=True)
    weights_file = 'weights/{}.pth'.format(name)
    torch.save(rnn.state_dict(), weights_file)


if __name__ == '__main__':
    main()
