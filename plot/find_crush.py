from pathlib import Path
import argparse
import json

def find_crashed_files(result_dir):
    result_dir = Path(result_dir)
    missing_hashes = []
    for json_path in result_dir.rglob('exp_dict.json'): #'./results'
        model_dir = json_path.parent
        p = model_dir / 'score_list.pkl'
        if p.exists():
            continue
        else:
            hash=p.parts[-2]
            print(hash)
            missing_hashes.append(hash)
    print(missing_hashes)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')
    parser.add_argument('--out-dir', type=Path, default='figures')
    args = parser.parse_args()
    print('works')
    find_crashed_files(args.result_dir)
    pass

if __name__=='__main__':
    main()