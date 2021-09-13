import sys

import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_scenario, show_sample_sequence
from rnn import (RNN, masked_mse_loss, SEQUENCE_STRIDE, RNN_HIDDEN_SIZE, 
    MLP_HIDDEN_SIZE, NUM_LAYERS)


def main():
    if len(sys.argv) != 3:
        print('Usage: python3 {} NAME SCENARIO'.format(sys.argv[0]))
        sys.exit(1)
    name = sys.argv[1]
    scenario = int(sys.argv[2])

    train_dataset, valid_dataset, test_dataset, train_stats = get_scenario(
        scenario,
        recurrent=True,
        sequence_stride=SEQUENCE_STRIDE)

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset
    mean_inputs, std_inputs, mean_targets, std_targets = train_stats

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
        train_preds, _ = rnn(train_inputs)
        valid_preds, _ = rnn(valid_inputs)
        test_preds, _ = rnn(test_inputs)

    mse_loss = masked_mse_loss(train_preds, train_targets, train_lengths,
                               train_lengths.max().item())
    print('Train MSE Loss: {:,.4f}'.format(mse_loss.item()))
    mse_loss = masked_mse_loss(valid_preds, valid_targets, valid_lengths,
                               valid_lengths.max().item())
    print('Valid MSE Loss: {:,.4f}'.format(mse_loss.item()))
    mse_loss = masked_mse_loss(test_preds, test_targets, test_lengths,
                               test_lengths.max().item())
    print('Test MSE Loss: {:,.4f}'.format(mse_loss.item()))

    show_sample_sequence(test_targets, test_preds, test_lengths, 
        recurrent=True)


if __name__ == '__main__':
    main()
