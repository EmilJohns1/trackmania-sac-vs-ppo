import argparse

from ppo_agent import train_ppo
from sac_agent import train_sac

def main(args):
    if args.agent == 'ppo':
        train_ppo()
    elif args.agent == 'sac':
        train_sac()
    else:
        raise Exception('Agent type not supported')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo')
    main(parser.parse_args())
