import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
parser.add_argument('--seq_length', type=int, default=32, help='sequence length for training')
parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')

parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

parser.add_argument('--n_trials', type=int, default=200, help='number of data sequences to collect in each episode')
parser.add_argument('--trial_len', type=int, default=256, help='number of steps in each trial')
parser.add_argument('--n_subseq', type=int, default=8, help='number of subsequences to divide each sequence into')
args = parser.parse_args()