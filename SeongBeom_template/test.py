import argparse
parser = argparse.ArgumentParser()

# Data and model checkpoints directories
parser.add_argument('--hi', type=int, default=42, help='random seed (default: 42)')
args = parser.parse_args()
print(args)
print(args.hi)