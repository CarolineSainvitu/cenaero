import sys

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_scenario, show_sample_sequence
from mlp import (MLP, SEQUENCE_STRIDE, HIDDEN_SIZE, NUM_HIDDEN)

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

    mlp = MLP(
        input_size=train_inputs.size(1),
        hidden_size=HIDDEN_SIZE,
        output_size=train_targets.size(1),
        num_hidden=NUM_HIDDEN,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    mlp.load_state_dict(torch.load('../weights/mlp-{}.pth'.format(name)))

    fig, ax = plt.subplots(figsize=(8, 6))
    losses = pd.read_csv('../results/mlp-{}.csv'.format(name), index_col=[0])
    losses.plot(ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        train_preds = mlp(train_inputs)
        valid_preds = mlp(valid_inputs)
        test_preds = mlp(test_inputs)

    mse_loss = F.mse_loss(train_preds, train_targets)
    print('Train MSE Loss: {:,.4f}'.format(mse_loss.item()))
    mse_loss = F.mse_loss(valid_preds, valid_targets)
    print('Valid MSE Loss: {:,.4f}'.format(mse_loss.item()))
    mse_loss = F.mse_loss(test_preds, test_targets)
    print('Test MSE Loss: {:,.4f}'.format(mse_loss.item()))

    show_sample_sequence(test_targets, test_preds, test_lengths, 
        recurrent=False)


if __name__ == '__main__':
    main()