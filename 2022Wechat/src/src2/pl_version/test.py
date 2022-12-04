import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, default=1)
parser.add_argument("--tasks", type=list, default=['mlm', 'cls'])

args = parser.parse_known_args()[0]


print(args.tasks)