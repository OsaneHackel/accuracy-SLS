from exp_configs import EXP_GROUPS
from runCode import exp_mnist, exp_cifar10, exp_cifar10_dense, exp_cifar100, exp_slope, exp_batch
import argparse
import os

from haven import haven_utils as hu

SWEEPS = {
    'mnist': exp_mnist,
    'cifar10': exp_cifar10,
    'cifar10-dense': exp_cifar10_dense,
    'cifar100': exp_cifar100,
    'slope': exp_slope,
    'batch': exp_batch,
}

def run_sbatch(exp_dicts, name, partition='gpu-2080ti', dry=False):
    hashes = []
    for exp_dict in exp_dicts:
        hashes += [f'"{hu.hash_dict(exp_dict)}"']
    cmd = f'sbatch -p {partition} --job-name {name} train.sbatch python run_job.py {" ".join(hashes)}'
    print(cmd)
    if not dry:
        os.system(cmd)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', choices=list(SWEEPS))
    parser.add_argument('-n', type=int, default=1, help='Number of experiments per job')
    parser.add_argument('-p', '--partition', default='gpu-2080ti', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    args = parser.parse_args()

    experiment_groups = SWEEPS[args.sweep]
    print(experiment_groups)
    exp_dicts = []
    counter = 0
    i = 1
    for exp_group in experiment_groups:
        print(exp_group)
        for exp in EXP_GROUPS[exp_group]:
            exp_dicts += [exp]
            counter += 1
            if counter == args.n:
                run_sbatch(exp_dicts, f'{args.sweep}-{i}', args.partition, args.dry_run)
                exp_dicts = []
                counter = 0
                i += 1

    if exp_dicts:
        run_sbatch(exp_dicts, args.partition)


if __name__ == '__main__':
    main()
