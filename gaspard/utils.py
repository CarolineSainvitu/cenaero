import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


DIRECTORIES_NAMES = {
    'initial': '38Q31TzlO',
    'b>': '38gYX2QrJ',
    'P>': '38gWjjw40',
    'additional': '38fO5rac1',
    'P<': '38gVDgZf9'
}

DATA_PATH = '../data/{}-{}/npz_data/data.npz'
PARAMS_PATH = '../data/{}-{}/Minamo_Parameters-Wall2D.txt'

POSITIONS = [
    (0.0, 10.0),
    (10.0, 10.0),
    (0.0, 6.0),
    (10.0, 6.0),
    (0.0, 2.0),
    (10.0, 2.0)
]


def load_data(datasets, recurrent=False, sequence_stride=10):

    inputs, targets = [], []

    for dataset in datasets:

        directory_name = DIRECTORIES_NAMES[dataset]
        simulation_id = 1

        data_path = DATA_PATH.format(directory_name, '{}')
        params_path = PARAMS_PATH.format(directory_name, '{}')

        while os.path.exists(data_path.format(simulation_id)):

            data = np.load(data_path.format(simulation_id))

            # Extract input data: `t`, `x_t`, `P_t`
            time = torch.from_numpy(data['time']).float()
            laser_position = torch.from_numpy(data['laser_position_x']).float()
            laser_power = torch.from_numpy(data['laser_power']).float()

            # Parse parameters
            with open(params_path.format(simulation_id)) as params_file:
                lines = params_file.read().splitlines()
                power = float(lines[0].split(' = ')[1])
                break_time = float(lines[1].split(' = ')[1])

            # Create features `P` and `b`
            power = torch.full(laser_power.shape, power)
            break_time = torch.full(laser_power.shape, break_time)

            for i, (x, y) in enumerate(POSITIONS):

                x_position = torch.full(laser_power.shape, x)
                y_position = torch.full(laser_power.shape, y)

                if recurrent:
                    # Create a feature `delta`
                    delta = time.clone()
                    delta[1:] = time[1:] - time[:-1]
                    input = torch.stack(
                        (delta, laser_position, laser_power, power, break_time,
                            x_position, y_position),
                        dim=1)
                else:
                    input = torch.stack(
                        (time, laser_position, laser_power, power, break_time,
                            x_position, y_position),
                        dim=1)

                # Extract target data: `P^1_t`, ..., `P^6_t`
                target = torch.from_numpy(data['T{}'.format(i + 1)]).float()
                target = target.unsqueeze(-1)

                input = input[::sequence_stride, :]
                target = target[::sequence_stride, :]

                inputs.append(input)
                targets.append(target)

            simulation_id += 1

        # Extract sequences lengths
        lengths = [len(input) for input in inputs]

    return inputs, targets, lengths


def split(sequences, splits, seed=20210831):
    torch.random.manual_seed(seed)
    num_sequences = len(sequences)
    permutation = torch.randperm(num_sequences)

    datasets = []
    start = None
    for split in splits + (None,):
        end = int(split * num_sequences) if split is not None else None
        datasets.append([sequences[i] for i in permutation[start:end]])
        start = end

    return datasets


def get_scenario(scenario, recurrent=False, sequence_stride=10):

    if scenario == 1:

        inputs, targets, lengths = load_data(['initial'],
                                             recurrent=False,
                                             sequence_stride=sequence_stride)

        splits = (0.7, 0.85)

        train_inputs, valid_inputs, test_inputs = split(inputs, splits)
        train_targets, valid_targets, test_targets = split(targets, splits)
        train_lengths, valid_lengths, test_lengths = split(lengths, splits)

    elif scenario == 2:

        inputs, targets, lengths = load_data(['initial'],
                                             recurrent=False,
                                             sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        inputs_test, targets_test, lengtths_test = load_data(
            ['additional'],
            recurrent=False,
            sequence_stride=sequence_stride)

    elif scenario == 3:

        inputs, targets, lengths = load_data(['initial'],
                                             recurrent=False,
                                             sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        inputs_test, targets_test, lengtths_test = load_data(
            ['P<', 'P>', 'b>'],
            recurrent=False,
            sequence_stride=sequence_stride)

    elif scenario == 4:

        inputs, targets, lengths = load_data(['initial', 'additional'],
                                             recurrent=False,
                                             sequence_stride=sequence_stride)

        splits = (0.7, 0.85)

        train_inputs, valid_inputs, test_inputs = split(inputs, splits)
        train_targets, valid_targets, test_targets = split(targets, splits)
        train_lengths, valid_lengths, test_lengths = split(lengths, splits)

    elif scenario == 5:

        inputs, targets, lengths = load_data(['initial', 'additional'],
                                             recurrent=False,
                                             sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        inputs_test, targets_test, lengtths_test = load_data(
            ['P<', 'P>', 'b>'],
            recurrent=False,
            sequence_stride=sequence_stride)

    else:
        raise ValueError('SCENARIO is not valid')

    if recurrent:
        train_inputs_flatten = torch.cat(train_inputs, dim=0)
        valid_inputs_flatten = torch.cat(valid_inputs, dim=0)
        test_inputs_flatten = torch.cat(test_inputs, dim=0)

        train_targets_flatten = torch.cat(train_targets, dim=0)
        valid_targets_flatten = torch.cat(valid_targets, dim=0)
        test_targets_flatten = torch.cat(test_targets, dim=0)

        mean_inputs = train_inputs_flatten.mean(dim=0)
        std_inputs = train_inputs_flatten.std(dim=0)
        mean_targets = train_targets_flatten.mean(dim=0)
        std_targets = train_targets_flatten.std(dim=0)

        train_inputs = nn.utils.rnn.pad_sequence(train_inputs)
        valid_inputs = nn.utils.rnn.pad_sequence(valid_inputs)
        test_inputs = nn.utils.rnn.pad_sequence(test_inputs)

        train_targets = nn.utils.rnn.pad_sequence(train_targets)
        valid_targets = nn.utils.rnn.pad_sequence(valid_targets)
        test_targets = nn.utils.rnn.pad_sequence(test_targets)
    else:
        train_inputs = torch.cat(train_inputs, dim=0)
        valid_inputs = torch.cat(valid_inputs, dim=0)
        test_inputs = torch.cat(test_inputs, dim=0)

        train_targets = torch.cat(train_targets, dim=0)
        valid_targets = torch.cat(valid_targets, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        mean_inputs = train_inputs.mean(dim=0)
        std_inputs = train_inputs.std(dim=0)
        mean_targets = train_targets.mean(dim=0)
        std_targets = train_targets.std(dim=0)

    train_lengths = torch.tensor(train_lengths)
    valid_lengths = torch.tensor(valid_lengths)
    test_lengths = torch.tensor(test_lengths)

    return ((train_inputs, train_targets, train_lengths),
            (valid_inputs, valid_targets, valid_lengths),
            (test_inputs, test_targets, test_lengths),
            (mean_inputs, std_inputs, mean_targets, std_targets))




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
