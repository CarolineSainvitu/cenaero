import sys

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from utils import get_scenario


NUM_SEQUENCES = 121
SEQUENCE_STRIDE = 10


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 {} SCENARIO'.format(sys.argv[0]))
        sys.exit(1)
    scenario = int(sys.argv[1])

    train_dataset, valid_dataset, test_dataset, train_stats = get_scenario(
        scenario,
        recurrent=False,
        sequence_stride=SEQUENCE_STRIDE)

    train_inputs, train_targets, train_seq_lengths = train_dataset
    valid_inputs, valid_targets, valid_seq_lengths = valid_dataset
    test_inputs, test_targets, test_seq_lengths = test_dataset

    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=None,
                               n_jobs=-1)

    if train_targets.size(1) == 1:
        train_targets = train_targets.squeeze(1)
        valid_targets = valid_targets.squeeze(1)
        test_targets = test_targets.squeeze(1)

    rf.fit(train_inputs, train_targets)
    test_preds = torch.from_numpy(rf.predict(test_inputs))

    mse_loss = F.mse_loss(test_preds, test_targets)
    print('MSE Loss: {:.4f}'.format(mse_loss))


if __name__ == '__main__':
    main()
