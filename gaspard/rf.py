import torch
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import utils


NUM_SEQUENCES = 121


def main():
    train_dataset, valid_dataset, test_dataset = utils.train_valid_test_split(
        range(1, NUM_SEQUENCES + 1), recurrent=False, train_ratio=0.7)

    train_inputs, train_targets = train_dataset
    valid_inputs, valid_targets = valid_dataset
    test_inputs, test_targets = test_dataset

    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=None,
                               n_jobs=-1)

    rf.fit(train_inputs, train_targets)
    test_preds = torch.from_numpy(rf.predict(test_inputs))

    mse_loss = ((test_preds - test_targets) ** 2).sum() / test_preds.shape[0]
    print('MSE Loss: {:.4f}'.format(mse_loss))


if __name__ == '__main__':
    main()
