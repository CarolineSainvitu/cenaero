import numpy as np
import matplotlib.pyplot as plt

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

        # TODO: delete this
        if recurrent:
            input = input[::20, :]
            target = target[::20, :]

        inputs.append(input)
        targets.append(target)

    # Extract sequences lengths
    seq_lengths = torch.tensor([len(input) for input in inputs])

    if recurrent:
        # Pad sequences and stack them to create the dataset
        inputs = nn.utils.rnn.pad_sequence(inputs)
        targets = nn.utils.rnn.pad_sequence(targets)

    else:
        # Concatenate all sequences to create the dataset
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

    return inputs, targets, seq_lengths


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


def show_sample_sequence(targets, preds, seq_lengths, recurrent=False):

    num_sequences = seq_lengths.size(0)
    sample_id = torch.randint(num_sequences, ())
    seq_length = seq_lengths[sample_id]

    if recurrent:
        target = targets[:seq_length, sample_id, :]
        pred = preds[:seq_length, sample_id, :]
    else:
        start_indices = seq_lengths.cumsum(dim=0)
        start_sequence = start_indices[sample_id]
        if sample_id == num_sequences - 1:
            end_sequence = None
        else:
            end_sequence = start_indices[sample_id + 1]
        target = targets[start_indices[sample_id]:end_sequence]
        pred = preds[start_indices[sample_id]:end_sequence]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(target.size(-1)):
        if i == 0:
            ax.plot(target[:, i], color='C{}'.format(i), label='Target')
            ax.plot(pred[:, i], ':', color='C{}'.format(i), label='Prediction')
        else:
            ax.plot(target[:, i], color='C{}'.format(i), label='_nolegend_')
            ax.plot(pred[:, i], ':', color='C{}'.format(i), label='_nolegend_')

    ax.legend()

    ax.set_xlabel('Time step [-]')
    ax.set_ylabel('Temperature [°C]')

    plt.tight_layout()
    plt.show()
