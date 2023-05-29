import os
import json
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetMACCSKeysFingerprint, GetMorganFingerprintAsBitVect


def feature_fp(modeling_folder, in_csv, in_sdf, mode='ECFP4-1024', trte_title='trte', target='target'):
    '''
    mode: 'MACCS', 'ECFP4-1024', 'PubChemFP-881', 'SubFP-307'
    '''
       
    df_init = pd.read_csv(in_csv)

    # calculate fingerprints
    if mode == 'ECFP4-1024':
        out_csv = modeling_folder + 'GroupB_ECFP4-1024_modeldf.csv'
        out_json = modeling_folder + 'GroupB_ECFP4-1024.json'
        des_df = cal_ECFP4(in_sdf)
        failed_ls = []
    
    elif mode == 'MACCS':
        out_csv = modeling_folder + 'GroupA_MACCS_modeldf.csv'
        out_json = modeling_folder + 'GroupA_MACCS.json'
        des_df = cal_MACCS(in_sdf)
        failed_ls = []
    
    elif mode == 'PubChemFP-881':
        out_csv = modeling_folder + 'GroupH_PubChemFP-881_modeldf.csv'
        out_json = modeling_folder + 'GroupH_PubChemFP-881.json'
        des_df, failed_ls = cal_PaDELFP(modeling_folder, in_sdf, mode='PubChemFP')
    
    elif mode == 'SubFP-307':
        out_csv = modeling_folder + 'GroupL_SubFP-307_modeldf.csv'
        out_json = modeling_folder + 'GroupL_SubFP-307.json'
        des_df, failed_ls = cal_PaDELFP(modeling_folder, in_sdf, mode='SubFP')

    else:
        print('error mode')
        return None

    if len(failed_ls) > 0:
        model_df = df_init.drop(failed_ls).reset_index(drop=True)
        model_df = pd.concat([model_df, des_df], axis=1)
    else:
        model_df = pd.concat([df_init, des_df], axis=1)
    
    model_df.to_csv(out_csv, index=None)

    # feature selection by IG
    des_dict = {mode + '_all': des_df.columns.tolist()}
    des_dict[mode + '_top128'] = IGfilter(model_df, trte_title, target, des_dict[mode + '_all'], 128)
    des_dict[mode + '_top64'] = IGfilter(model_df, trte_title, target, des_dict[mode + '_all'], 64)
    save_json(des_dict, out_json)

    print('cal_descriptors: {} done.'.format(mode))


def cal_MACCS(sdf_file):

    suppl = Chem.SDMolSupplier(sdf_file)
    name_ls = ['MACCS_' + str(i) for i in range(0, 167)]
    fp_ar = np.zeros(shape=(len(suppl), 167))
    
    for i, mol in enumerate(suppl):
        fp = GetMACCSKeysFingerprint(mol)
        fp_ar[i] = fp
    
    # save
    df = pd.DataFrame(fp_ar, columns=name_ls)

    return df


def cal_ECFP4(sdf_file):

    suppl = Chem.SDMolSupplier(sdf_file)
    name_ls = ['ECFP4_{}'.format(i) for i in range(1, 1025)]
    fp_ar = np.zeros(shape=(len(suppl), 1024))
    
    for i, mol in enumerate(suppl):
        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_ar[i] = fp
    
    # save
    df = pd.DataFrame(fp_ar, columns=name_ls)

    return df

def cal_PaDELFP(modeling_folder, sdf_file, mode='PubChemFP'):
    '''
    mode: 'PubChemFP' or 'SubFP'
    '''
    if mode == 'PubChemFP':
        out_csv = modeling_folder + 'GroupH_PubChemFP-881_modeldf.csv'
        out_txt = modeling_folder + 'GroupH_PubChemFP-881_failedID.txt'
    elif mode == 'SubFP':
        out_csv = modeling_folder + 'GroupL_SubFP-307_modeldf.csv'
        out_txt = modeling_folder + 'GroupL_SubFP-307_failedID.txt'
    else:
        return None
    
    # calculate, obtain no-ordered, with errors -> csv
    subprocess.call(['java', '-jar', '/home/cadd/tools/PaDEL-Descriptor/PaDEL-Descriptor.jar',
                    '-fingerprints',
                    '-threads', '1',
                    '-descriptortypes', '/home/cadd/tools/PaDEL-Descriptor/descriptors_{}.xml'.format(mode),
                    '-detectaromaticity', '-standardizenitro',
                    '-maxruntime', '120000',
                    '-dir', sdf_file,
                    '-file', out_csv[:-4] + '_cal.csv'])
    
    # order
    df = pd.read_csv(out_csv[:-4] + '_cal.csv')
    
    if df.shape[0] > 1:
        idx = []

        for name in df.loc[:, 'Name'].values:
            idx.append(int(name.split('_')[-1]))

        df = pd.concat([pd.Series(idx, name='TEMPID'), df], axis=1)
        df = df.sort_values(by='TEMPID').reset_index(drop=True)
        df = df.iloc[:, 1:]
    
    df.to_csv(out_csv[:-4] + '_ordered.csv', index=None)

    # remove errors
    failed_ls = []
    with open(out_csv, 'w') as fw:
        with open(out_csv[:-4] + '_ordered.csv') as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    fw.write(line)
                else:
                    if (',,' in line) or (',\n' in line) or ('Inf' in line) or ('inf' in line):
                        failed_ls.append(i-1)
                        continue
                    else:
                        fw.write(line)
    
    if len(failed_ls) > 0:
        with open(out_txt, 'w') as f:
            for idx in failed_ls:
                f.write(str(idx) + '\n')
    
    df = pd.read_csv(out_csv).iloc[:, 1:]

    if mode == 'PubChemFP':
        df.columns = ['PubChemFP_{}'.format(i) for i in range(1, 882)]
    elif mode == 'SubFP':
        df.columns = ['SubFP_{}'.format(i) for i in range(1, 308)]
    else:
        pass
    
    os.remove(out_csv)
    os.remove(out_csv[:-4] + '_cal.csv')
    os.remove(out_csv[:-4] + '_ordered.csv')

    return df, failed_ls


def cal_entropy(X):
    '''Calculate information entropy of 1D array X, i.e., H(X)
    '''
    unique_ls = set([X[i] for i in range(X.shape[0])])
    
    ent = 0
    for value in unique_ls:
        p = X[X == value].shape[0] / X.shape[0]
        logp = np.log2(p)
        ent -= p  * logp
    return ent

def cal_conditional_entropy(X, Y):
    '''Calculate conditional entropy of given 1D arrays X and Y, i.e., H(Y|X)
    '''
    uniqueX_ls = set([X[i] for i in range(X.shape[0])])
    
    cond_ent = 0
    for value in uniqueX_ls:
        subset_Y = Y[X == value]
        p = subset_Y.shape[0] / Y.shape[0]
        cond_ent += p * cal_entropy(subset_Y)
    return cond_ent

def cal_infogain(X, Y):
    '''Calculate information gain of X and Y
    usually used for cal_infogain(fp, label)
    '''
    result = cal_entropy(Y) - cal_conditional_entropy(X, Y)
    return result

def IGfilter(model_df, trte_title, label_title, des_ls, keep_bits):
    '''
    Example:
    >>> select_des = IGfilter(model_df, 'trte', 'target', des_dict['ECFP4-1024_all'], keep_bits=128)
    '''
    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', des_ls].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    
    ig_ar = np.zeros(shape=len(des_ls))
    for i in range(len(des_ls)):
        ig_ar[i] = cal_infogain(tr_X[:, i], tr_Y)
    
    keep_id = np.argsort(-ig_ar)[:keep_bits]
    select_des = list(np.array(des_ls)[keep_id])

    return select_des

def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))