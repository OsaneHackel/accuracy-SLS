from exp_configs import EXP_GROUPS
from pathlib import Path
import argparse
from haven import haven_utils as hu

batches=['same', 'small', 'big']
def give_batch_hashes():
    MISSING_HASHES =[]
    E={}
    E['mnist_batch']=EXP_GROUPS['mnist_batch']
    E['cifar10_batch']=EXP_GROUPS['cifar10_batch']
    E['cifar100_batch']=EXP_GROUPS['cifar100_batch']
    E['cifar10_batch_dense']=EXP_GROUPS['cifar10_batch_dense']
    for exp_group in E.values():
        for experiment in exp_group:
            exp_hash = hu.hash_dict(experiment)
            MISSING_HASHES.append(exp_hash)
    print(MISSING_HASHES)
    return(MISSING_HASHES)

def check_if_exists(missing, result_dir):
    result_dir = Path(result_dir)
    for hash in missing:
        p= result_dir/ hash
        if p.exists():
            print(f'{hash} exists')
            missing.remove(hash)
    print(missing)

def all_missing_hashes(result_dir):
    missing_hashes = []
    result_dir=Path(result_dir)
    for exp in EXP_GROUPS.values():
        for dct in exp:
            hash = hu.hash_dict(dct)
            p = result_dir/hash
            if not p.exists():
                print(f'{hash} does not exist')
                missing_hashes.append(hash)
    print(missing_hashes)
    return(missing_hashes)

def compare_lists(missing, all_missing):
    for hash in all_missing:
        if hash not in missing:
            print(f'{hash} is not in missing')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')   
    args = parser.parse_args()
    #missing=give_batch_hashes()
    #check_if_exists(missing, args.result_dir)
    #all_missing=all_missing_hashes(args.result_dir)
    #compare_lists(missing, all_missing)
    print(EXP_GROUPS['cifar10_sls'])
    pass
if __name__=='__main__':
    main()