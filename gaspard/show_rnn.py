import sys

import torch
import pandas as pd
import matplotlib.pyplot as plt

import utils
from rnn import (RNN, masked_mse_loss, NUM_SEQUENCES, SEQUENCE_STRIDE,
    BATCH_SIZE, RNN_HIDDEN_SIZE, MLP_HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE,
    NUM_EPOCH_CONVERGENCE)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 {} NAME'.format(sys.argv[0]))
        sys.exit(1)
    name = sys.argv[1]

    train_dataset, valid_dataset, test_dataset = utils.train_valid_test_split(
        range(1, NUM_SEQUENCES + 1), recurrent=True, train_ratio=0.7,
        sequence_stride=SEQUENCE_STRIDE)

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset

    # Standardization statistics
    train_max_length = train_lengths.max().item()
    num_train = train_inputs.size(1)
    input_size = train_inputs.size(-1)
    output_size = train_targets.size(-1)
    timesteps = torch.arange(train_max_length).expand(num_train,
                                                      train_max_length)
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
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        output_size=train_targets.size(2),
        num_layers=NUM_LAYERS,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    rnn.load_state_dict(torch.load('../weights/rnn-{}.pth'.format(name)))

    fig, ax = plt.subplots(figsize=(8, 6))
    losses = pd.read_csv('../results/rnn-{}.csv'.format(name), index_col=[0])
    losses.plot(ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        test_preds, _ = rnn(test_inputs)

    mse_loss = masked_mse_loss(test_preds, test_targets, test_lengths,
                               test_lengths.max().item())
    print('Test MSE Loss: {:,.4f}'.format(mse_loss.item()))

    while True:
        utils.show_sample_sequence(
            test_targets, test_preds, test_lengths, recurrent=True)


if __name__ == '__main__':
    main()
