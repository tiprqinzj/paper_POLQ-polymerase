import os, json, time, math
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import GraphDescriptors, Descriptors, MolSurf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


def feature_svr_numeric(modeling_folder, in_csv, in_sdf, mode='PaDEL2D-784', trte_title='trte', target='target',
                        min_des_num=4, max_des_num=64, step_des_num=4):
    '''
    mode: ''
    '''
       
    df_init = pd.read_csv(in_csv)

    # calculate fingerprints
    if mode == 'PaDEL2D-784':
        out_csv = modeling_folder + 'GroupP_PaDEL2D-784_modeldf.csv'
        out_json = modeling_folder + 'GroupP_PaDEL2D-784.json'
        sfs_log = modeling_folder + 'GroupP_PaDEL2D-784_sfs.log'
        des_df, failed_ls = cal_PaDEL2D(modeling_folder, in_sdf)
    
    elif mode == 'RDKit2D-208':
        out_csv = modeling_folder + 'GroupR_RDKit2D-208_modeldf.csv'
        out_json = modeling_folder + 'GroupR_RDKit2D-208.json'
        sfs_log = modeling_folder + 'GroupR_RDKit2D-208_sfs.log'
        des_df, failed_ls = cal_RDKit2D(modeling_folder, in_sdf)

    else:
        print('error mode')
        return None

    if len(failed_ls) > 0:
        model_df = df_init.drop(failed_ls).reset_index(drop=True)
        model_df = pd.concat([model_df, des_df], axis=1)
    else:
        model_df = pd.concat([df_init, des_df], axis=1)
    
    model_df.to_csv(out_csv, index=None)

    # feature selection
    des_dict = {mode + '_all': des_df.columns.tolist()}
    des_dict[mode + '_PCC'] = PCCfilter(model_df, trte_title, target, des_dict[mode + '_all'])
    des_dict = SFSrank_SVR(model_df, trte_title, target, des_dict, mode, sfs_log, min_des_num, max_des_num, step_des_num)
    save_json(des_dict, out_json)


    print('cal_descriptors: {} done.'.format(mode))


def feature_rfc_numeric(modeling_folder, in_csv, in_sdf, mode='PaDEL2D-784', trte_title='trte', target='target',
                        min_des_num=4, max_des_num=64, step_des_num=4):
       
    df_init = pd.read_csv(in_csv)

    # calculate fingerprints
    if mode == 'PaDEL2D-784':
        out_csv = modeling_folder + 'GroupP_PaDEL2D-784_modeldf.csv'
        out_json = modeling_folder + 'GroupP_PaDEL2D-784.json'
        sfs_log = modeling_folder + 'GroupP_PaDEL2D-784_sfs.log'
        des_df, failed_ls = cal_PaDEL2D(modeling_folder, in_sdf)
    
    elif mode == 'RDKit2D-208':
        out_csv = modeling_folder + 'GroupR_RDKit2D-208_modeldf.csv'
        out_json = modeling_folder + 'GroupR_RDKit2D-208.json'
        sfs_log = modeling_folder + 'GroupR_RDKit2D-208_sfs.log'
        des_df, failed_ls = cal_RDKit2D(modeling_folder, in_sdf)

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
    des_dict[mode + '_PCC'] = PCCfilter(model_df, trte_title, target, des_dict[mode + '_all'])
    des_dict[mode + '_RFECV5'] = RFECVfilter_RFC(model_df, trte_title, target, des_dict[mode + '_PCC'], 5)
    des_dict = SFSrank_RFC(model_df, trte_title, target, des_dict, mode + '_RFECV5', sfs_log, min_des_num, max_des_num, step_des_num)
    save_json(des_dict, out_json)

    print('cal_descriptors: {} done.'.format(mode))


def RFECVfilter_RFC(model_df, trte_title, label_title, select_des, FOLD=5):

    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', select_des].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    tr_N = tr_Y.shape[0]

    scaler = StandardScaler()
    tr_X = scaler.fit_transform(tr_X)

    if tr_N >= 2000:
        n_trees = 100
    elif tr_N <= 200:
        n_trees = 10
    else:
        n_trees = int(tr_N / 20)
    
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    selector = RFECV(clf, cv=StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=42), n_jobs=8)
    
    selector.fit(tr_X, tr_Y)
    ranking_ = selector.ranking_

    des_ar = np.array(select_des)
    result_des = list(des_ar[np.where(ranking_ == 1)[0]])
    
    return result_des


def cal_RDKit2D(modeling_folder, sdf_file):

    out_txt = modeling_folder + 'GroupR_RDKit2D-208_failedID.txt'

    suppl = Chem.SDMolSupplier(sdf_file)
    name_ls = _get_RDKit2D_name()
    des_ar = np.zeros(shape=(len(suppl), len(name_ls)))
    
    for i, mol in enumerate(suppl):
        des_ls = _cal_RDKit2D(mol)
        des_ar[i] = np.array(des_ls)
    
    df = pd.DataFrame(des_ar, columns=name_ls)

    failed_ls = []
    for i in range(df.shape[0]):
        if df.loc[i].isna().sum() > 0:
            failed_ls.append(i)
    
    if len(failed_ls) > 0:
        df = df.drop(index=failed_ls).reset_index(drop=True)
        with open(out_txt, 'w') as f:
            for idx in failed_ls:
                f.write(str(idx) + '\n')
    
    return df, failed_ls

def cal_PaDEL2D(modeling_folder, sdf_file):

    out_csv = modeling_folder + 'GroupP_PaDEL2D-784_modeldf.csv'
    out_txt = modeling_folder + 'GroupP_PaDEL2D-784_failedID.txt'
    
    # calculate, obtain no-ordered, with errors -> csv
    subprocess.call(['java', '-jar', '/home/cadd/tools/PaDEL-Descriptor/PaDEL-Descriptor.jar',
                    '-2d',
                    '-threads', '1',
                    '-descriptortypes', '/home/cadd/tools/PaDEL-Descriptor/descriptors_PaDEL2D.xml',
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
        df = df.iloc[:, 2:]
    else:
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
    
    df = pd.read_csv(out_csv)
    
    os.remove(out_csv)
    os.remove(out_csv[:-4] + '_cal.csv')
    os.remove(out_csv[:-4] + '_ordered.csv')

    return df, failed_ls


def _get_RDKit2D_name():
    
    name_trad2D = [
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
        'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'EState_VSA11',
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
        'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10',
        'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
        'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
        'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
        'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
        'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
        'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'ExactMolWt', 'MolWt', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'TPSA', 'Ipc', 'qed',
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha',
        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
        'HeavyAtomCount', 'RingCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons',
        'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
        'NumValenceElectrons'
    ]

    name_fr = [
        'fr_Al_COO',
        'fr_Al_OH',
        'fr_Al_OH_noTert',
        'fr_ArN',
        'fr_Ar_COO',
        'fr_Ar_N',
        'fr_Ar_NH',
        'fr_Ar_OH',
        'fr_COO',
        'fr_COO2',
        'fr_C_O',
        'fr_C_O_noCOO',
        'fr_C_S',
        'fr_HOCCN',
        'fr_Imine',
        'fr_NH0',
        'fr_NH1',
        'fr_NH2',
        'fr_N_O',
        'fr_Ndealkylation1',
        'fr_Ndealkylation2',
        'fr_Nhpyrrole',
        'fr_SH',
        'fr_aldehyde',
        'fr_alkyl_carbamate',
        'fr_alkyl_halide',
        'fr_allylic_oxid',
        'fr_amide',
        'fr_amidine',
        'fr_aniline',
        'fr_aryl_methyl',
        'fr_azide',
        'fr_azo',
        'fr_barbitur',
        'fr_benzene',
        'fr_benzodiazepine',
        'fr_bicyclic',
        'fr_diazo',
        'fr_dihydropyridine',
        'fr_epoxide',
        'fr_ester',
        'fr_ether',
        'fr_furan',
        'fr_guanido',
        'fr_halogen',
        'fr_hdrzine',
        'fr_hdrzone',
        'fr_imidazole',
        'fr_imide',
        'fr_isocyan',
        'fr_isothiocyan',
        'fr_ketone',
        'fr_ketone_Topliss',
        'fr_lactam',
        'fr_lactone',
        'fr_methoxy',
        'fr_morpholine',
        'fr_nitrile',
        'fr_nitro',
        'fr_nitro_arom',
        'fr_nitro_arom_nonortho',
        'fr_nitroso',
        'fr_oxazole',
        'fr_oxime',
        'fr_para_hydroxylation',
        'fr_phenol',
        'fr_phenol_noOrthoHbond',
        'fr_phos_acid',
        'fr_phos_ester',
        'fr_piperdine',
        'fr_piperzine',
        'fr_priamide',
        'fr_prisulfonamd',
        'fr_pyridine',
        'fr_quatN',
        'fr_sulfide',
        'fr_sulfonamd',
        'fr_sulfone',
        'fr_term_acetylene',
        'fr_tetrazole',
        'fr_thiazole',
        'fr_thiocyan',
        'fr_thiophene',
        'fr_unbrch_alkane',
        'fr_urea'
    ]
    
    return name_trad2D + name_fr


def _cal_RDKit2D(mol):
    
    BCUT_ls = [
        Descriptors.BCUT2D_MWHI(mol), Descriptors.BCUT2D_MWLOW(mol),
        Descriptors.BCUT2D_CHGHI(mol), Descriptors.BCUT2D_CHGLO(mol),
        Descriptors.BCUT2D_LOGPHI(mol), Descriptors.BCUT2D_LOGPLOW(mol),
        Descriptors.BCUT2D_MRHI(mol), Descriptors.BCUT2D_MRLOW(mol)
    ]

    Graph_ls = [
        GraphDescriptors.BalabanJ(mol), GraphDescriptors.BertzCT(mol),
        GraphDescriptors.Chi0(mol), GraphDescriptors.Chi0n(mol), GraphDescriptors.Chi0v(mol),
        GraphDescriptors.Chi1(mol), GraphDescriptors.Chi1n(mol), GraphDescriptors.Chi1v(mol),
        GraphDescriptors.Chi2n(mol), GraphDescriptors.Chi2v(mol),
        GraphDescriptors.Chi3n(mol), GraphDescriptors.Chi3v(mol),
        GraphDescriptors.Chi4n(mol), GraphDescriptors.Chi4v(mol)
    ]

    EState_ls = [
        Descriptors.EState_VSA1(mol), Descriptors.EState_VSA2(mol), 
        Descriptors.EState_VSA3(mol), Descriptors.EState_VSA4(mol),
        Descriptors.EState_VSA5(mol), Descriptors.EState_VSA6(mol),
        Descriptors.EState_VSA7(mol), Descriptors.EState_VSA8(mol),
        Descriptors.EState_VSA9(mol), Descriptors.EState_VSA10(mol),
        Descriptors.EState_VSA11(mol),
        Descriptors.VSA_EState1(mol), Descriptors.VSA_EState2(mol),
        Descriptors.VSA_EState3(mol), Descriptors.VSA_EState4(mol),
        Descriptors.VSA_EState5(mol), Descriptors.VSA_EState6(mol),
        Descriptors.VSA_EState7(mol), Descriptors.VSA_EState8(mol),
        Descriptors.VSA_EState9(mol), Descriptors.VSA_EState10(mol)
    ]
    MolSurf_ls = [
        MolSurf.PEOE_VSA1(mol), MolSurf.PEOE_VSA2(mol), MolSurf.PEOE_VSA3(mol),
        MolSurf.PEOE_VSA4(mol), MolSurf.PEOE_VSA5(mol), MolSurf.PEOE_VSA6(mol),
        MolSurf.PEOE_VSA7(mol), MolSurf.PEOE_VSA8(mol), MolSurf.PEOE_VSA9(mol),
        MolSurf.PEOE_VSA10(mol), MolSurf.PEOE_VSA11(mol), MolSurf.PEOE_VSA12(mol),
        MolSurf.PEOE_VSA13(mol), MolSurf.PEOE_VSA14(mol),
        MolSurf.SMR_VSA1(mol), MolSurf.SMR_VSA2(mol), MolSurf.SMR_VSA3(mol),
        MolSurf.SMR_VSA4(mol), MolSurf.SMR_VSA5(mol), MolSurf.SMR_VSA6(mol),
        MolSurf.SMR_VSA7(mol), MolSurf.SMR_VSA8(mol), MolSurf.SMR_VSA9(mol),
        MolSurf.SMR_VSA10(mol),
        MolSurf.SlogP_VSA1(mol), MolSurf.SlogP_VSA2(mol), MolSurf.SlogP_VSA3(mol),
        MolSurf.SlogP_VSA4(mol), MolSurf.SlogP_VSA5(mol), MolSurf.SlogP_VSA6(mol),
        MolSurf.SlogP_VSA7(mol), MolSurf.SlogP_VSA8(mol), MolSurf.SlogP_VSA9(mol),
        MolSurf.SlogP_VSA10(mol), MolSurf.SlogP_VSA11(mol), MolSurf.SlogP_VSA12(mol),
    ]

    Property_ls = [
        Descriptors.ExactMolWt(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.TPSA(mol),
        Descriptors.Ipc(mol),
        Descriptors.qed(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.MaxAbsEStateIndex(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MaxEStateIndex(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinAbsEStateIndex(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.MinEStateIndex(mol),
        Descriptors.MinPartialCharge(mol)
    ]
    
    Count_ls = [
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAliphaticCarbocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticCarbocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumValenceElectrons(mol)
    ]

    fr_ls = [
        Descriptors.fr_Al_COO(mol),
        Descriptors.fr_Al_OH(mol),
        Descriptors.fr_Al_OH_noTert(mol),
        Descriptors.fr_ArN(mol),
        Descriptors.fr_Ar_COO(mol),
        Descriptors.fr_Ar_N(mol),
        Descriptors.fr_Ar_NH(mol),
        Descriptors.fr_Ar_OH(mol),
        Descriptors.fr_COO(mol),
        Descriptors.fr_COO2(mol),
        Descriptors.fr_C_O(mol),
        Descriptors.fr_C_O_noCOO(mol),
        Descriptors.fr_C_S(mol),
        Descriptors.fr_HOCCN(mol),
        Descriptors.fr_Imine(mol),
        Descriptors.fr_NH0(mol),
        Descriptors.fr_NH1(mol),
        Descriptors.fr_NH2(mol),
        Descriptors.fr_N_O(mol),
        Descriptors.fr_Ndealkylation1(mol),
        Descriptors.fr_Ndealkylation2(mol),
        Descriptors.fr_Nhpyrrole(mol),
        Descriptors.fr_SH(mol),
        Descriptors.fr_aldehyde(mol),
        Descriptors.fr_alkyl_carbamate(mol),
        Descriptors.fr_alkyl_halide(mol),
        Descriptors.fr_allylic_oxid(mol),
        Descriptors.fr_amide(mol),
        Descriptors.fr_amidine(mol),
        Descriptors.fr_aniline(mol),
        Descriptors.fr_aryl_methyl(mol),
        Descriptors.fr_azide(mol),
        Descriptors.fr_azo(mol),
        Descriptors.fr_barbitur(mol),
        Descriptors.fr_benzene(mol),
        Descriptors.fr_benzodiazepine(mol),
        Descriptors.fr_bicyclic(mol),
        Descriptors.fr_diazo(mol),
        Descriptors.fr_dihydropyridine(mol),
        Descriptors.fr_epoxide(mol),
        Descriptors.fr_ester(mol),
        Descriptors.fr_ether(mol),
        Descriptors.fr_furan(mol),
        Descriptors.fr_guanido(mol),
        Descriptors.fr_halogen(mol),
        Descriptors.fr_hdrzine(mol),
        Descriptors.fr_hdrzone(mol),
        Descriptors.fr_imidazole(mol),
        Descriptors.fr_imide(mol),
        Descriptors.fr_isocyan(mol),
        Descriptors.fr_isothiocyan(mol),
        Descriptors.fr_ketone(mol),
        Descriptors.fr_ketone_Topliss(mol),
        Descriptors.fr_lactam(mol),
        Descriptors.fr_lactone(mol),
        Descriptors.fr_methoxy(mol),
        Descriptors.fr_morpholine(mol),
        Descriptors.fr_nitrile(mol),
        Descriptors.fr_nitro(mol),
        Descriptors.fr_nitro_arom(mol),
        Descriptors.fr_nitro_arom_nonortho(mol),
        Descriptors.fr_nitroso(mol),
        Descriptors.fr_oxazole(mol),
        Descriptors.fr_oxime(mol),
        Descriptors.fr_para_hydroxylation(mol),
        Descriptors.fr_phenol(mol),
        Descriptors.fr_phenol_noOrthoHbond(mol),
        Descriptors.fr_phos_acid(mol),
        Descriptors.fr_phos_ester(mol),
        Descriptors.fr_piperdine(mol),
        Descriptors.fr_piperzine(mol),
        Descriptors.fr_priamide(mol),
        Descriptors.fr_prisulfonamd(mol),
        Descriptors.fr_pyridine(mol),
        Descriptors.fr_quatN(mol),
        Descriptors.fr_sulfide(mol),
        Descriptors.fr_sulfonamd(mol),
        Descriptors.fr_sulfone(mol),
        Descriptors.fr_term_acetylene(mol),
        Descriptors.fr_tetrazole(mol),
        Descriptors.fr_thiazole(mol),
        Descriptors.fr_thiocyan(mol),
        Descriptors.fr_thiophene(mol),
        Descriptors.fr_unbrch_alkane(mol),
        Descriptors.fr_urea(mol),
    ]
    
    des_ls = BCUT_ls + Graph_ls + EState_ls + MolSurf_ls + Property_ls + Count_ls + fr_ls
    
    return des_ls


def PCCfilter(model_df, trte_title, label_title, des_ls, inner_threshold=0.85):
    '''
    Example:
    remain_des = PCCfilter(model_df, 'trte', 'target', des_ls)
    '''
    print('Initial descriptor number: {}'.format(len(des_ls)))
    
    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', des_ls].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    print('training samples: {}'.format(tr_X.shape[0]))

    # scaler
    scaler = StandardScaler()
    tr_scaleX = scaler.fit_transform(tr_X)
    
    del_by_std = []

    for i, des in enumerate(des_ls):
        std = tr_scaleX[:, i].std()
        if std == 0:
            del_by_std.append(des)
    print('Total of {} descritors delete because of STD == 0'.format(len(del_by_std)))
    
    # Corr with label
    corr_ls = []

    for i, des in enumerate(des_ls):
        if des not in del_by_std:
            corr = np.abs(np.corrcoef(tr_scaleX[:, i], tr_Y)[0, 1])
            corr_ls.append(corr)

    corr_sr = pd.Series(corr_ls, index=[des for des in des_ls if des not in del_by_std])
    corr_sr = corr_sr.sort_values(ascending=False)
    
    # filter with inner-correlation
    del_by_inner = []

    for i, desi in enumerate(corr_sr.index):
        if desi in del_by_inner:
            continue
        else:
            for j in range(i+1, len(corr_sr)):
                desj = corr_sr.index[j]
                if desj in del_by_inner:
                    continue
                else:
                    indexi = des_ls.index(desi)
                    indexj = des_ls.index(desj)
                    corr = np.abs(np.corrcoef(tr_scaleX[:, indexi], tr_scaleX[:, indexj])[0, 1])
                    if corr >= inner_threshold:
                        del_by_inner.append(desj)
    print('Total of {} descritors delete because of inner-corr >= {}'.format(len(del_by_inner), inner_threshold))
    
    remain_des = []
    for des in des_ls:
        if des not in del_by_std and des not in del_by_inner:
            remain_des.append(des)

    # rank by corrlation-with-label
    remain_des = [des for des in corr_sr.index.tolist() if des in remain_des]
    print('After STD and CORR filtering, remain descritors: {}'.format(len(remain_des)))
    
    return remain_des


def SFSrank_SVR(model_df, trte_title, label_title, des_dict, mode, logfile, min_des_num, max_des_num, step_des_num):

    trte_ar = model_df.loc[:, trte_title].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values

    # base estimator
    regr = SVR()

    des_ls = des_dict[mode + '_PCC']
    print('Start SFS calculation: start {}'.format(len(des_ls)))

    with open(logfile, 'w') as f:
        f.write('Start SFS calculation\n')
        f.write('descriptor number: init {}\n\n'.format(len(des_ls)))

    t = time.time()
    scaler = StandardScaler()
    tr_X = scaler.fit_transform(model_df.loc[trte_ar == 'train', des_ls].values)

    
    for des_num in range(min_des_num, max_des_num + 1, step_des_num):

        if des_num >= len(des_ls):
            break

        selector = SequentialFeatureSelector(regr,
                                             n_features_to_select = des_num,
                                             direction='forward',
                                             scoring='neg_root_mean_squared_error',
                                             n_jobs=8)
        
        selector.fit(tr_X, tr_Y)
        select_des = np.array(des_ls)[selector.get_support()].tolist()
        des_dict['{}_SFSforward{}'.format(mode, des_num)] = select_des

        with open(logfile, 'a') as f:
            f.write('forward {}, time {}, des = {}\n'.format(len(select_des), cal_time(t), ', '.join(select_des)))

    return des_dict


def SFSrank_RFC(model_df, trte_title, label_title, des_dict, des_title, logfile, min_des_num, max_des_num, step_des_num):

    trte_ar = model_df.loc[:, trte_title].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    tr_N = tr_Y.shape[0]

    # base estimator
    if tr_N >= 2000:
        n_trees = 100
    elif tr_N <= 200:
        n_trees = 10
    else:
        n_trees = int(tr_N / 20)
    
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    des_ls = des_dict[des_title]
    print('Start SFS calculation: start {}'.format(len(des_ls)))

    with open(logfile, 'w') as f:
        f.write('Start SFS calculation\n')
        f.write('descriptor number: init {}\n\n'.format(len(des_ls)))

    t = time.time()
    scaler = StandardScaler()
    tr_X = scaler.fit_transform(model_df.loc[trte_ar == 'train', des_ls].values)

    for des_num in range(min_des_num, max_des_num + 1, step_des_num):

        if des_num >= len(des_ls):
            break

        selector = SequentialFeatureSelector(clf,
                                             n_features_to_select = des_num,
                                             direction='forward',
                                             scoring='balanced_accuracy',
                                             n_jobs=8)

        selector.fit(tr_X, tr_Y)
        select_des = np.array(des_ls)[selector.get_support()].tolist()
        des_dict['{}_SFSforward{}'.format(des_title, des_num)] = select_des

        with open(logfile, 'a') as f:
            f.write('forward {}, time {}, des = {}\n'.format(len(select_des), cal_time(t), ', '.join(select_des)))
    
    return des_dict


def cal_time(since):
    now = time.time()
    s = now - since

    if s > 3600:
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s = s - h * 3600 - m * 60
        out = '{}h {}m {:.0f}s'.format(h, m, s)
    else:
        m = math.floor(s / 60)
        s = s - m * 60
        out = '{}m {:.0f}s'.format(m, s)
    return out


def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))
    

if __name__ == '__main__':


    # manual data set: cal_descriptor only

    folder = '/home/cadd/Project/Manual_DataSet/Kinase_Inhibitors_221107/'
    in_csv = '/home/cadd/Project/Manual_DataSet/Kinase_Inhibitors_221107/database_curated92.csv'
    smi_title = 'Checked SMILES'
    out_sdf = '/home/cadd/Project/Manual_DataSet/Kinase_Inhibitors_221107/database_curated92.sdf'

    # load data
    df = pd.read_csv(in_csv)

    # save sdf
    string_molblock = ''
    for smi in df.loc[:, smi_title].values:

        mol = Chem.MolFromSmiles(smi)
        molblock = Chem.MolToMolBlock(mol)
        string_molblock += molblock + '$$$$\n'

    with open(out_sdf, 'w') as f:
        f.write(string_molblock)
    
    # cal RDKit2D
    df_rdkit2d, failed_rdkit2d = cal_RDKit2D(folder, out_sdf)
    df_padel2d, failed_padel2d = cal_PaDEL2D(folder, out_sdf)

    # save
    df_rdkit2d.to_csv(folder + 'RDKit2D-208.csv', index=None)
    df_padel2d.to_csv(folder + 'PaDEL2D-784.csv', index=None)

    if len(failed_rdkit2d) > 0:
        with open(folder + 'RDKit2D-208_failedID.txt', 'w') as f:
            for idx in failed_rdkit2d:
                f.write(str(idx) + '\n')
    
    if len(failed_padel2d) > 0:
        with open(folder + 'PaDEL2D-784_failedID.txt', 'w') as f:
            for idx in failed_padel2d:
                f.write(str(idx) + '\n')



    