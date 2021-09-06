import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

import utils

from copy import deepcopy


NUM_SEQUENCES = 121

BATCH_SIZE = 16
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCH_CONVERGENCE = 5


class RNN(nn.Module):
    def __init__(self, cell, input_size, hidden_size, output_size, num_layers,
            mean_inputs, std_inputs, mean_targets, std_targets):
        super().__init__()

        self.mean_inputs = mean_inputs
        self.std_inputs = std_inputs
        self.mean_targets = mean_targets
        self.std_targets = std_targets

        if cell == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers)
        elif cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers)
        else:
            raise NotImplementedError

        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
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
            loss = masked_mse_loss(batch_preds, batch_targets, batch_lengths,
                                   train_max_length)

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
            train_after_epoch = masked_mse_loss(train_preds, train_targets,
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

    rnn.load_state_dict(best_weights)

    return {
        'train': train_losses,
        'train_after': train_after_epoch_losses,
        'valid': valid_losses
    }


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 {} NAME'.format(sys.argv[0]))
        sys.exit(1)
    name = sys.argv[1]

    train_dataset, valid_dataset, test_dataset = utils.train_valid_test_split(
        range(1, NUM_SEQUENCES + 1), recurrent=True, train_ratio=0.7)

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset

    # Standardization statistics
    train_max_length = train_lengths.max().item()
    num_train = train_inputs.size(1)
    input_size = train_inputs.size(-1)
    output_size = train_targets.size(-1)
    timesteps = torch.arange(train_max_length).expand(num_train, train_max_length)
    masks = (timesteps < train_lengths.unsqueeze(1)).T.unsqueeze(-1)

    train_inputs_flatten = train_inputs[masks.expand(-1, -1, input_size)]
    train_inputs_flatten = train_inputs_flatten.view(-1, input_size)
    train_targets_flatten = train_targets[masks.expand(-1, -1, output_size)]
    train_targets_flatten = train_targets_flatten.view(-1, output_size)

    mean_inputs = train_inputs_flatten.mean(dim=0)
    std_inputs = train_inputs_flatten.std(dim=0)
    mean_targets = train_targets_flatten.mean(dim=0)
    std_targets = train_targets_flatten.std(dim=0)

    rnn = RNN(
        cell='gru',
        input_size=train_inputs.size(2),
        hidden_size=HIDDEN_SIZE,
        output_size=train_targets.size(2),
        num_layers=NUM_LAYERS,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    opt = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

    stats = train_rnn(rnn, opt, train_inputs, train_targets, train_lengths,
        valid_inputs, valid_targets, valid_lengths, BATCH_SIZE,
        NUM_EPOCH_CONVERGENCE)

    os.makedirs('../results/', exist_ok=True)
    df = pd.DataFrame.from_dict(stats)
    df.to_csv('../results/rnn-{}.csv'.format(name))

    with torch.no_grad():
        test_preds, _ = rnn(test_inputs)

    utils.show_sample_sequence(
        test_targets, test_preds, test_lengths, recurrent=True)

    mse_loss = masked_mse_loss(test_preds, test_targets, test_lengths,
                               test_lengths.max().item())
    print('Test MSE Loss: {:,.4f}'.format(mse_loss.item()))

    os.makedirs('../weights/', exist_ok=True)
    weights_file = '../weights/rnn-{}.pth'.format(name)
    torch.save(rnn.state_dict(), weights_file)


if __name__ == '__main__':
    main()
