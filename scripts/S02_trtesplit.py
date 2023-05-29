import sys
sys.path.append('/home/cadd/Project/tipr_coding/')

import time
import pandas as pd

from utils import trtesplit_by_TSNE, cal_time, save_sdf

def main(curated_file, out_file, smi_title, target='target', trte_title='trte', radio=5):

    '''radio=5 means 4:1'''

    print('Start calculation ...\n')
    time_start = time.time()

    df = pd.read_csv(curated_file)
    df_trte, _ = trtesplit_by_TSNE(df, smi_title, trte_title, target, radio)

    df_trte.to_csv(out_file, index=None)

    # save SDF file
    sdf_file = curated_file[:-4] + '.sdf'
    save_sdf(df, smi_title, sdf_file)

    print('Result file was saved at: {}'.format(out_file))
    print('SDF file was saved at:    {}'.format(sdf_file))

    print('\nCalculation done. Time {}\n'.format(cal_time(time_start)))



if __name__ == '__main__':

    main(
        curated_file = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/datasetV5beta3_Caco2_Perm10_curated826.csv',
        out_file     = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/datasetV5beta3_Caco2_Perm10_curated826_trte.csv',
        smi_title    = 'Flatten SMILES',
        target       = 'target_binary10',
        trte_title   = 'trte',
        radio        = 5,
    )

