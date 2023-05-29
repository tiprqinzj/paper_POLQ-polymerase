import sys

if sys.platform == 'linux':
    sys.path.append('/home/cadd/Project/tipr_coding/')
if sys.platform == 'darwin':
    sys.path.append('/Users/qinzijian/Experiment/Project/tipr_coding_macbook/')

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import cal_time

def main(curated_file, out_file, id_title='NO', target='target', trte_title='trte', validnum=5):
    print('Start calculation ...\n')

    time_start = time.time()

    df = pd.read_csv(curated_file)
    tr_df = df.loc[df.loc[:, trte_title].values == 'train'].reset_index(drop=True)
    te_df = df.loc[df.loc[:, trte_title].values == 'test'].reset_index(drop=True)
    
    N = tr_df.shape[0]
    X = np.random.random(N)
    Y = tr_df.loc[:, target].values

    result_ls = [''] * N
    skf = StratifiedKFold(n_splits=validnum, shuffle=True, random_state=42)

    for i, (_, index) in enumerate(skf.split(X, Y)):
        name = 'train' + str(i+1)
        for j in index:
            result_ls[j] = name
    
    result_ls += ['test'] * te_df.shape[0]
    
    sr = pd.Series(result_ls, name='trtecv')

    df_out = pd.concat([tr_df, te_df], axis=0).reset_index(drop=True)
    df_out = pd.concat([df_out, sr], axis=1)
    df_out = df_out.sort_values(by=id_title).reset_index(drop=True)

    df_out.to_csv(out_file, index=None)

    print('Result file was saved at: {}'.format(out_file))

    print('\nCalculation done. Time {}\n'.format(cal_time(time_start)))



if __name__ == '__main__':

    main(
        curated_file = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/datasetV5beta3_Caco2_Perm10_curated826_trte.csv',
        out_file     = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/datasetV5beta3_Caco2_Perm10_curated826_trtecv.csv',
        id_title     = 'NO',
        target       = 'target_binary10',
        trte_title   = 'trte',
        validnum     = 5,
    )
