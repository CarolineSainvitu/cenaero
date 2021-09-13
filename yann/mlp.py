import os
import sys

from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from utils import load_data, split, get_scenario

from copy import deepcopy


SEQUENCE_STRIDE = 10

BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_HIDDEN = 2
LEARNING_RATE = 1e-3
NUM_EPOCH_CONVERGENCE = 8


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden,
            mean_inputs, std_inputs, mean_targets, std_targets):
        super().__init__()

        self.mean_inputs = mean_inputs
        self.std_inputs = std_inputs
        self.mean_targets = mean_targets
        self.std_targets = std_targets

        hidden_layers = []
        for _ in range(num_hidden - 1):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = (x - self.mean_inputs) / self.std_inputs
        x = self.sequential(x)
        x = x * self.std_targets + self.mean_targets
        return x


def train_mlp(mlp, opt, train_inputs, train_targets, valid_inputs,
        valid_targets, batch_size, num_epoch_convergence):

    lowest_loss, num_epoch_no_improvement = float('inf'), 0
    best_weights = deepcopy(mlp.state_dict())
    train_losses, valid_losses = [], []
    train_after_epoch_losses = []  # TODO: delete this

    num_train = train_targets.size(0)

    epoch = 0
    while num_epoch_no_improvement < num_epoch_convergence:

        try:
            # Training
            permutation = torch.randperm(num_train)
            train_loss = 0.0

            for i in range(0, num_train, batch_size):
                indices = permutation[i:i+batch_size]
                batch_inputs = train_inputs[indices, :]
                batch_targets = train_targets[indices, :]

                batch_preds = mlp(batch_inputs)
                loss = F.mse_loss(batch_preds, batch_targets)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item()

            train_loss /= int(num_train / batch_size)
            train_losses.append(train_loss)

            epoch += 1

            # Training evaluation after epoch
            with torch.no_grad():
                train_preds = mlp(train_inputs)
                train_after_epoch = F.mse_loss(train_preds, train_targets).item()

            train_after_epoch_losses.append(train_after_epoch)

            # Validation
            with torch.no_grad():
                valid_preds = mlp(valid_inputs)
                valid_loss = F.mse_loss(valid_preds, valid_targets).item()

            valid_losses.append(valid_loss)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                num_epoch_no_improvement = 0
                best_weights = deepcopy(mlp.state_dict())
            else:
                num_epoch_no_improvement += 1

            print('Epoch {:03d}'.format(epoch))
            print('\tTrain: {:.4f}'.format(train_loss))
            print('\tTrain after epoch: {:.4f}'.format(train_after_epoch))
            print('\tValid: {:.4f}'.format(valid_loss))

        except KeyboardInterrupt:
            if len(train_after_epoch_losses) < len(train_losses):
                train_after_epoch_losses.append(None)
            if len(valid_losses) < len(train_losses):
                valid_losses.append(None)
            print()
            break

    mlp.load_state_dict(best_weights)

    return {
        'train': train_losses,
        'train_after': train_after_epoch_losses,
        'valid': valid_losses
    }


def main():
    if len(sys.argv) != 3:
        print('Usage: python3 {} NAME SCENARIO'.format(sys.argv[0]))
        sys.exit(1)
    name = sys.argv[1]
    scenario = int(sys.argv[2])

    train_dataset, valid_dataset, test_dataset, train_stats = get_scenario(
        scenario,
        recurrent=False,
        sequence_stride=SEQUENCE_STRIDE)

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset
    mean_inputs, std_inputs, mean_targets, std_targets = train_stats

    print('Training size: {}'.format(train_inputs.size(0)))
    print('Validation size: {}'.format(valid_inputs.size(0)))
    print('Test size: {}'.format(test_inputs.size(0)))

    mlp = MLP(
        input_size=train_inputs.size(1),
        hidden_size=HIDDEN_SIZE,
        output_size=train_targets.size(1),
        num_hidden=NUM_HIDDEN,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    opt = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

    start = perf_counter()
    stats = train_mlp(mlp, opt, train_inputs, train_targets, valid_inputs,
        valid_targets, BATCH_SIZE, NUM_EPOCH_CONVERGENCE)
    end = perf_counter()

    os.makedirs('../results/', exist_ok=True)
    df = pd.DataFrame.from_dict(stats)
    df.to_csv('../results/mlp-{}.csv'.format(name))

    print('Total CPU time {:.2f}'.format(end - start))
    print('CPU time per epoch {:.2f}'.format((end - start) / len(df)))

    with torch.no_grad():
        test_preds = mlp(test_inputs)

    mse_loss = F.mse_loss(test_preds, test_targets)
    print('Test MSE Loss: {:.4f}'.format(mse_loss.item()))

    os.makedirs('../weights/', exist_ok=True)
    weights_file = '../weights/mlp-{}.pth'.format(name)
    torch.save(mlp.state_dict(), weights_file)


if __name__ == '__main__':
    main()
