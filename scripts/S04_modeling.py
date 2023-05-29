from cal_fingerprint import feature_fp
from cal_descriptor import feature_rfc_numeric
from modeling_rfc_fp import build_rfc_fp_model
from modeling_rfc_num import build_rfc_num_model
from utils_public import extract_metrics_clf


def pipeline(modeling_folder, in_csv, in_sdf, trte='trte', trtecv='trtecv', target_title='target_binary',
             min_des_num=4, max_des_num=64, step_des_num=4):

    feature_fp(modeling_folder, in_csv, in_sdf, 'MACCS', trte, target_title)
    feature_fp(modeling_folder, in_csv, in_sdf, 'ECFP4-1024', trte, target_title)
    feature_fp(modeling_folder, in_csv, in_sdf, 'PubChemFP-881', trte, target_title)
    feature_fp(modeling_folder, in_csv, in_sdf, 'SubFP-307', trte, target_title)

    for des, prefix in zip(['MACCS_all', 'MACCS_top128', 'MACCS_top64'], ['ModelA1', 'ModelA2', 'ModelA3']):
        build_rfc_fp_model(modeling_folder + 'GroupA_MACCS_modeldf.csv', modeling_folder + 'GroupA_MACCS.json',
                           trte, trtecv, target_title, des, prefix, 5)
    
    for des, prefix in zip(['ECFP4-1024_all', 'ECFP4-1024_top128', 'ECFP4-1024_top64'], ['ModelB1', 'ModelB2', 'ModelB3']):
        build_rfc_fp_model(modeling_folder + 'GroupB_ECFP4-1024_modeldf.csv', modeling_folder + 'GroupB_ECFP4-1024.json',
                           trte, trtecv, target_title, des, prefix, 5)
    
    for des, prefix in zip(['PubChemFP-881_all', 'PubChemFP-881_top128', 'PubChemFP-881_top64'], ['ModelH1', 'ModelH2', 'ModelH3']):
        build_rfc_fp_model(modeling_folder + 'GroupH_PubChemFP-881_modeldf.csv', modeling_folder + 'GroupH_PubChemFP-881.json',
                           trte, trtecv, target_title, des, prefix, 5)
    
    for des, prefix in zip(['SubFP-307_all', 'SubFP-307_top128', 'SubFP-307_top64'], ['ModelL1', 'ModelL2', 'ModelL3']):
        build_rfc_fp_model(modeling_folder + 'GroupL_SubFP-307_modeldf.csv', modeling_folder + 'GroupL_SubFP-307.json',
                           trte, trtecv, target_title, des, prefix, 5)
    
    feature_rfc_numeric(modeling_folder, in_csv, in_sdf, 'PaDEL2D-784', trte, target_title, min_des_num, max_des_num, step_des_num)
    feature_rfc_numeric(modeling_folder, in_csv, in_sdf, 'RDKit2D-208', trte, target_title, min_des_num, max_des_num, step_des_num)

    start = 1
    for n in range(min_des_num, max_des_num + 1, step_des_num):

        build_rfc_num_model(modeling_folder + 'GroupP_PaDEL2D-784_modeldf.csv',
                            modeling_folder + 'GroupP_PaDEL2D-784.json',
                            trte, trtecv, target_title,
                            'PaDEL2D-784_RFECV5_SFSforward{}'.format(n), 'ModelP{}'.format(start), 5)
        start += 1
    
    start = 1
    for n in range(min_des_num, max_des_num + 1, step_des_num):

        build_rfc_num_model(modeling_folder + 'GroupR_RDKit2D-208_modeldf.csv',
                            modeling_folder + 'GroupR_RDKit2D-208.json',
                            trte, trtecv, target_title,
                            'RDKit2D-208_RFECV5_SFSforward{}'.format(n), 'ModelR{}'.format(start), 5)
        start += 1

    extract_metrics_clf(modeling_folder)



if __name__ == '__main__':

    FOLDER = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/'

    pipeline(
        modeling_folder = FOLDER,
        in_csv = FOLDER + 'datasetV5beta3_Caco2_Perm10_curated826_trtecv.csv',
        in_sdf = FOLDER + 'datasetV5beta3_Caco2_Perm10_curated826.sdf',
        trte = 'trte', trtecv='trtecv', 
        target_title = 'target_binary10',
        min_des_num = 4,
        max_des_num = 64,
        step_des_num = 2,
    )
