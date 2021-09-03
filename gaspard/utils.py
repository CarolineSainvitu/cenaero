import numpy as np

import torch
import torch.nn as nn


DATA_PATH = '../data/38Q31TzlO-{}/npz_data/data.npz'
PARAMS_PATH = '../data/38Q31TzlO-{}/Minamo_Parameters-Wall2D.txt'


def load_data(simulation_ids, recurrent=False):

    inputs, targets = [], []

    for simulation_id in simulation_ids:

        data = np.load(DATA_PATH.format(simulation_id))

        # Extract input data: `t`, `x_t`, `P_t`
        time = torch.from_numpy(data['time']).float()
        laser_position = torch.from_numpy(data['laser_position_x']).float()
        laser_power = torch.from_numpy(data['laser_power']).float()

        # Parse parameters
        with open(PARAMS_PATH.format(simulation_id)) as params_file:
            lines = params_file.read().splitlines()
            power = float(lines[0].split(' = ')[1])
            break_time = float(lines[1].split(' = ')[1])

        if recurrent:
            # Create a feature `delta`
            delta = time.clone()
            delta[1:] = time[1:] - time[:-1]
            input = torch.stack((delta, laser_position, laser_power), dim=1)
        else:
            # Create features `P` and `b`
            power = torch.full(laser_power.shape, power)
            break_time = torch.full(laser_power.shape, break_time)

            input = torch.stack(
                (time, laser_position, laser_power, power, break_time),
                dim=1)

        # Extract target data: `P^1_t`, ..., `P^6_t`
        target = torch.stack(
            [torch.from_numpy(data['T{}'.format(i + 1)]).float()
                for i in range(6)],
            dim=1)

        inputs.append(input)
        targets.append(target)

    if recurrent:
        # Extract sequences lengths
        seq_lengths = torch.tensor([len(input) for input in inputs])

        # Pad sequences and stack them to create the dataset
        inputs = nn.utils.rnn.pad_sequence(inputs)
        outputs = nn.utils.rnn.pad_sequence(inputs)
        inputs = torch.stack(inputs, dim=1)
        targets = torch.stack(targets, dim=1)

        return inputs, targets, seq_lengths

    else:
        # Concatenate all sequences to create the dataset
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        return inputs, targets


def train_valid_test_split(sequence_ids, recurrent=False, train_ratio=0.7,
        seed=20210831):

    # Shuffle sequences with a random seed
    torch.random.manual_seed(seed)
    n_sequences = len(sequence_ids)
    permutation = torch.randperm(n_sequences)
    shuffled_ids = torch.tensor(sequence_ids)[permutation]

    # Split sequences
    end_train = int(train_ratio * n_sequences)
    end_valid = int((train_ratio + 0.5 * (1.0 - train_ratio)) * n_sequences)

    train_ids = shuffled_ids[:end_train]
    valid_ids = shuffled_ids[end_train:end_valid]
    test_ids = shuffled_ids[end_valid:]

    train_dataset = load_data(train_ids, recurrent)
    valid_dataset = load_data(valid_ids, recurrent)
    test_dataset = load_data(test_ids, recurrent)

    return train_dataset, valid_dataset, test_dataset
