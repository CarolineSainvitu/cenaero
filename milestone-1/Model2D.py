import os
from time import perf_counter
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE=", device)
torch.backends.cudnn.deterministic = True


##########################################################
# Data loading
##########################################################

DIRECTORIES_NAMES = {
    'initial': '38Q31TzlO',
    'b>': '38gYX2QrJ',
    'P>': '38gWjjw40',
    'additional': '38fO5rac1',
    'P<': '38gVDgZf9'
}

DATASETS_DIR = '/home/lsalesse/Documents/Datasets/ARIAC-AM-DataBase/Tier0/'

DATASETS_DIRECTORIES_NAMES = {
    'initial': 'DataBase',
    'b>': 'DataBaseBTgt10',
    'P>': 'DataBasePgt250',
    'additional': 'DataBaseExtra',
    'P<': 'DataBasePlt50'
}


DATA_PATH = DATASETS_DIR+'{}/{}-{}/npz_data/data.npz'
PARAMS_PATH = DATASETS_DIR+'{}/{}-{}/Minamo_Parameters-Wall2D.txt'

def load_data(datasets, sequence_stride=10):

    inputs, targets = [], []

    for dataset in datasets:

        directory_name = DIRECTORIES_NAMES[dataset]
        datasetDirName = DATASETS_DIRECTORIES_NAMES[dataset]
        simulation_id = 1

        data_path = DATA_PATH.format(datasetDirName, directory_name, '{}')
        params_path = PARAMS_PATH.format(datasetDirName, directory_name, '{}')

        while os.path.exists(data_path.format(simulation_id)):
            print("Open simulation ", simulation_id)
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

            input = torch.stack(
                (
                    time,
                    laser_position,
                    laser_power,
                    power,
                    break_time
                ),
                dim=1
            )

            # Extract target data: `P^1_t`, ..., `P^6_t`
            target = torch.from_numpy(data['temperatures']).float()
            #target = target.unsqueeze(-1)

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

def get_scenario(scenario, sequence_stride=10):

    if scenario == 1:

        inputs, targets, lengths = load_data(['initial'], sequence_stride=sequence_stride)

        splits = (0.7, 0.85)

        train_inputs, valid_inputs, test_inputs = split(inputs, splits)
        train_targets, valid_targets, test_targets = split(targets, splits)
        train_lengths, valid_lengths, test_lengths = split(lengths, splits)

    elif scenario == 2:

        inputs, targets, lengths = load_data(['initial'], sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        test_inputs, test_targets, test_lengths = load_data(
            ['additional'],sequence_stride=sequence_stride)

        test_inputs, = split(test_inputs, ())
        test_targets, = split(test_targets, ())
        test_lengths, = split(test_lengths, ())

    elif scenario == 3:

        inputs, targets, lengths = load_data(['initial'], sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        test_inputs, test_targets, test_lengths = load_data(
            ['P<', 'P>', 'b>'],sequence_stride=sequence_stride)

        test_inputs, = split(test_inputs, ())
        test_targets, = split(test_targets, ())
        test_lengths, = split(test_lengths, ())

    elif scenario == 4:

        inputs, targets, lengths = load_data(['initial', 'additional'],sequence_stride=sequence_stride)

        splits = (0.7, 0.85)

        train_inputs, valid_inputs, test_inputs = split(inputs, splits)
        train_targets, valid_targets, test_targets = split(targets, splits)
        train_lengths, valid_lengths, test_lengths = split(lengths, splits)

    elif scenario == 5:

        inputs, targets, lengths = load_data(['initial', 'additional'],sequence_stride=sequence_stride)

        splits = (0.8,)

        train_inputs, valid_inputs = split(inputs, splits)
        train_targets, valid_targets = split(targets, splits)
        train_lengths, valid_lengths = split(lengths, splits)

        test_inputs, test_targets, test_lengths = load_data(
            ['P<', 'P>', 'b>'],sequence_stride=sequence_stride)

        test_inputs, = split(test_inputs, ())
        test_targets, = split(test_targets, ())
        test_lengths, = split(test_lengths, ())

    else:
        raise ValueError('SCENARIO is not valid')

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

##########################################################
# Model
##########################################################
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view([x.size(0)] + self.shape)

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden,
            mean_inputs, std_inputs, mean_targets, std_targets):
        super().__init__()

        self.mean_inputs = mean_inputs.to(device)
        self.std_inputs = std_inputs.to(device)
        self.mean_targets = mean_targets.to(device).unsqueeze(0)
        self.std_targets = std_targets.to(device).unsqueeze(0)

        hidden_layers = []
        for _ in range(num_hidden - 1):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.GELU())

        C = 10
        convInputShape = [C, 7, 13]
        convInputSize = C*7*13

        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            *hidden_layers,
            nn.Linear(hidden_size, convInputSize),
            Reshape(convInputShape),
            nn.ConvTranspose2d(
                in_channels=C,
                out_channels=C,
                kernel_size=(3, 4),
                stride=(1, 2),
                padding=(1, 1),
                dilation=1,
            ),  # 7x13 -> 7x26
            nn.GELU(),
            #nn.BatchNorm2d(C),
            nn.ConvTranspose2d(
                in_channels=C,
                out_channels=C,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=1,
            ),  # 7x26 -> 13x51
            nn.GELU(),
            #nn.BatchNorm2d(C),
            nn.ConvTranspose2d(
                in_channels=C,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=1,
            ),  # 13x51 -> 25x101
            #nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = (x - self.mean_inputs) / self.std_inputs
        x = self.sequential(x)
        x = x * self.std_targets + self.mean_targets
        x = x.squeeze(1)
        return x


##########################################################
# Training
##########################################################
def train(model, opt, train_inputs, train_targets, valid_inputs,
        valid_targets, batch_size, num_epoch_convergence):
    train_inputs = train_inputs.to(device)
    valid_inputs = valid_inputs.to(device)
    train_targets = train_targets.to(device)
    valid_targets = valid_targets.to(device)
    model.to(device)
    lowest_loss, num_epoch_no_improvement = float('inf'), 0
    best_weights = deepcopy(model.state_dict())
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

                batch_preds = model(batch_inputs)
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
                train_preds = model(train_inputs)
                train_after_epoch = F.mse_loss(train_preds, train_targets).item()

            train_after_epoch_losses.append(train_after_epoch)

            # Validation
            with torch.no_grad():
                valid_preds = model(valid_inputs)
                valid_loss = F.mse_loss(valid_preds, valid_targets).item()

            valid_losses.append(valid_loss)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                num_epoch_no_improvement = 0
                best_weights = deepcopy(model.state_dict())
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

    model.load_state_dict(best_weights)

    return {
        'train': train_losses,
        'train_after': train_after_epoch_losses,
        'valid': valid_losses
    }

if __name__ == '__main__':
    scenario = 1
    SEQUENCE_STRIDE = 20
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    NUM_EPOCH_CONVERGENCE = 8
    train_dataset, valid_dataset, test_dataset, train_stats = get_scenario(
        scenario,
        sequence_stride=SEQUENCE_STRIDE
    )

    train_inputs, train_targets, train_lengths = train_dataset
    valid_inputs, valid_targets, valid_lengths = valid_dataset
    test_inputs, test_targets, test_lengths = test_dataset
    mean_inputs, std_inputs, mean_targets, std_targets = train_stats

    print('Training size: {}'.format(train_inputs.size(0)))
    print('Validation size: {}'.format(valid_inputs.size(0)))
    print('Test size: {}'.format(test_inputs.size(0)))

    model = CNNModel(
        input_size=train_inputs.size(1),
        hidden_size=128,
        output_size=train_targets.size(1),
        num_hidden=4,
        mean_inputs=mean_inputs,
        std_inputs=std_inputs,
        mean_targets=mean_targets,
        std_targets=std_targets)

    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = perf_counter()
    stats = train(model, opt, train_inputs, train_targets, valid_inputs,
        valid_targets, BATCH_SIZE, NUM_EPOCH_CONVERGENCE)
    end = perf_counter()

    #os.makedirs('results/', exist_ok=True)
    #df = pd.DataFrame.from_dict(stats)
    #df.to_csv('results/{}.csv'.format(name))

    print('Total GPU time {:.2f}'.format(end - start))
    #print('GPU time per epoch {:.2f}'.format((end - start) / len(df)))

    with torch.no_grad():
        test_inputs = test_inputs.to(device)
        test_preds = model(test_inputs)

    mse_loss = F.mse_loss(test_preds, test_targets.to(device))
    print('Test MSE Loss: {:.4f}'.format(mse_loss.item()))

    test_preds_ = test_preds.to('cpu')
    test_targets_ = test_targets.to('cpu')
    test_inputs_ = test_inputs.to("cpu")
    relative_Error = torch.linalg.norm(test_preds_ - test_targets_) / torch.linalg.norm(test_targets_)
    print("L2 relative error=", relative_Error.numpy())
    print("Linf relative error", ((test_preds_ - test_targets_).max() / (test_targets_).max()).numpy())

    for sample in range(test_targets_.shape[0])[:100]:
        print("Generate plot for sample ", sample)
        plt.clf()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.set_size_inches(10, 10)
        fig.set_dpi(100)
        R = test_targets_[sample]
        im1 = ax1.imshow(R, vmax=R.max(), vmin=R.min())
        ax1.set_title("Reference")
        fig.colorbar(im1, ax=ax1)
        P = test_preds_[sample]
        im2 = ax2.imshow(P, vmax=R.max(), vmin=R.min())
        ax2.set_title("Prediction")
        fig.colorbar(im2, ax=ax2)
        im3 = ax3.imshow(np.square(P - R))
        ax3.set_title("Erreur L2")
        fig.colorbar(im3, ax=ax3)
        im4 = ax4.imshow(np.abs(P - R))
        ax4.set_title("Erreur L1")
        fig.colorbar(im4, ax=ax4)
        figname = str(sample)  # "" test_inputs_[sample]
        plt.savefig("./2DFigures_2/" + figname + ".png")
        plt.close(fig)

