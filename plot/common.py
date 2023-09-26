from pathlib import Path
import json
import pandas as pd
import numpy as np


def give_score_lists(result_dir, dataset, model, optimizer, slope = None, version = None, batch=None, weight_decay=None, epoch=200, run=None):
    result_dir = Path(result_dir)
    print('give_score_lists')
    score_lists = []
    path = Path('results')
    for json_path in result_dir.rglob('exp_dict.json'): #'./results'
        model_dir = json_path.parent
        score_list_path = model_dir / 'score_list.pkl'

        with open(json_path, 'r') as f:
            exp_dict = json.load(f)

        exp_model = exp_dict['model']
        exp_optimizer = exp_dict['opt']['name']
        exp_dataset = exp_dict['dataset']
        
        if exp_dict['max_epoch'] != epoch:
            continue

        if exp_dataset != dataset:
            continue

        if exp_model != model:
            continue

        if exp_optimizer != optimizer:
            continue

        if version is not None:
            #print( version)
            #print(exp_dict['version'])
            exp_version=exp_dict['version']
            if exp_version != version:
                continue 

        if batch is not None:
            #print(batch)
            #print(exp_dict['searchBatch'])
            exp_batch = exp_dict['searchBatch'] 
            if exp_batch != batch:
                continue
        
        if slope is not None: 
            exp_slope = exp_dict['slope for weak accurracy']
            print(exp_slope)
            #print(str(exp_slope) != slope)
            if str(exp_slope) != slope:
                continue
        if weight_decay is not None:
            exp_weight_decay = exp_dict['opt']['weight_decay']
            if exp_weight_decay != weight_decay:
                continue
        if run is not None:
            exp_run = exp_dict['runs']
            if exp_run != run:
                continue
        print('found', json_path)
        score_list = pd.read_pickle(score_list_path)
        score_lists.append(score_list)
    return score_lists


def give_values(df, value):
    #print(df)
    epoch = np.unique([item['epoch'] for item in df])
    list = np.zeros(len(epoch))
    for i, item in enumerate(df):
        list[i] = item[value]
    return list

def give_exp_dict(result_dir, dataset, model, optimizer, slope = None, version = None, batch=None, weight_decay=None, epoch=200):
    result_dir = Path(result_dir)
    print('give_score_lists')
    exp_dicts = []
    for json_path in result_dir.rglob('exp_dict.json'): 
        with open(json_path, 'r') as f:
            exp_dict = json.load(f)

        exp_model = exp_dict['model']
        exp_optimizer = exp_dict['opt']['name']
        exp_dataset = exp_dict['dataset']

        if exp_dict['max_epoch'] != epoch:
            continue

        if exp_dataset != dataset:
            continue

        if exp_model != model:
            continue

        if exp_optimizer != optimizer:
            continue

        if version is not None:
            exp_version=exp_dict['version']
            if exp_version != version:
                continue 

        if batch is not None:
            #print(batch)
            #print(exp_dict['searchBatch'])
            exp_batch = exp_dict['searchBatch'] 
            if exp_batch != batch:
                continue
        
        if slope is not None: 
            exp_slope = exp_dict['slope for weak accurracy']
            #print(type(slope))
            # print(type(exp_slope))
            #print(str(exp_slope) != slope)
            if str(exp_slope) != slope:
                continue
        if weight_decay is not None:
            exp_weight_decay = exp_dict['opt']['weight_decay']
            if exp_weight_decay != weight_decay:
                continue
        
        print('found', json_path)   
        exp_dicts.append(exp_dict)
    return exp_dicts

def main():
    pass

if __name__ == '__main__':
    main()