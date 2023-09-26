import argparse
from pathlib import Path
from exp_configs import EXP_GROUPS
import trainval

from haven import haven_utils as hu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hashes', nargs='+')
    parser.add_argument('--data', type=Path, default='data')
    parser.add_argument('--results', type=Path, default='results')
    args = parser.parse_args()

    exp_lookup = {}
    for exp_group in EXP_GROUPS.values():
        for experiment in exp_group:
            exp_hash = hu.hash_dict(experiment)
            exp_lookup[exp_hash] = experiment

    experiments = []
    for hash_ in args.hashes:
        try:
            experiment = exp_lookup[hash_]
            experiments.append(experiment)
        except KeyError:
            raise ValueError(f'Unknown experiment hash: {hash_}')

    for i, exp in enumerate(experiments):
        print(f'Executing experiment #{i}: {args.hashes[i]}')

        trainval.trainval(
            exp, 
            str(args.results), 
            str(args.data),
            reset=False, 
            metrics_flag=1, 
            DerivTest=False,
            epsilon=1e-5, 
            runtime=False, 
            normalization=False,
        )
    


if __name__ == '__main__':
    main()