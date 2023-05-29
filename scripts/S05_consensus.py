import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from matplotlib.ticker import PercentFormatter


def cal_score_conf(ar, avgY):
    mean = ar[~np.isnan(ar)].mean()
    std  = ar[~np.isnan(ar)].std()

    if np.isnan(mean) or np.isnan(std):
        return np.nan, np.nan
        
    else:
        if std == 0:
            conf = 0.0
        else:
            conf = min(
                scipy.stats.norm(mean, std).cdf(avgY),
                1 - scipy.stats.norm(mean, std).cdf(avgY)
            )
        
        return round(mean, 3), round(conf, 3)


def model_clffolder(folder, df_des, failed_ls=[], scale=True, validnum=5):

    # load data
    clfs = [joblib.load(folder + 'train{}.model'.format(i)) for i in range(1, validnum+1)]
    with open(folder + 'model_des.txt') as f:
        model_des = [eachline.strip() for eachline in f]
    
    if scale:
        scaler = joblib.load(folder + 'build.scaler')
        te_X = scaler.transform(df_des.loc[:, model_des].values)
    else:
        te_X = df_des.loc[:, model_des].values
    
    # predict
    te_N = te_X.shape[0] + len(failed_ls)
    te_score = np.zeros(shape=(te_N, validnum))

    for i, clf in enumerate(clfs):

        score = clf.predict_proba(te_X)[:, 1]

        if len(failed_ls) > 0:
            for idx in failed_ls:
                score = np.insert(score, idx, np.nan)
        
        te_score[:, i] = score
    
    return te_score


def Metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try: se = float(tp) / float(tp + fn)
    except ZeroDivisionError: se = -1.0
    try: sp = float(tn) / float(tn + fp)
    except ZeroDivisionError: sp = -1.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return tp, tn, fp, fn, se, sp, acc, mcc

def clf_prod_ABHLPR(df_maccs, df_ecfp4,
                    df_pubchemfp, failed_pubchemfp, df_subfp, failed_subfp,
                    df_padel2d, failed_padel2d, df_rdkit2d, failed_rdkit2d):

    FOLDER = '/home/cadd/Project/Permeability_Caco2/Caco2_Perm10_modelV5beta3_prodRFC_ABHLPR/'
    tr_avgY = 0.586

    folders = [
        FOLDER + 'ModelA2_MACCS_top128_RFC/',
        FOLDER + 'ModelB3_ECFP4-1024_top64_RFC/',
        FOLDER + 'ModelH2_PubChemFP-881_top128_RFC/',
        FOLDER + 'ModelL3_SubFP-307_top64_RFC/',
        FOLDER + 'ModelP7_PaDEL2D-784_RFECV5_SFSforward16_RFC/',
        FOLDER + 'ModelR8_RDKit2D-208_RFECV5_SFSforward18_RFC/',
    ]
    
    te_pred = np.zeros(shape=(df_maccs.shape[0], 30))

    te_pred[:, 0:5]   = model_clffolder(folders[0], df_maccs, failed_ls=[], scale=False, validnum=5)
    te_pred[:, 5:10]  = model_clffolder(folders[1], df_ecfp4, failed_ls=[], scale=False, validnum=5)
    te_pred[:, 10:15] = model_clffolder(folders[2], df_pubchemfp, failed_ls=failed_pubchemfp, scale=False, validnum=5)
    te_pred[:, 15:20] = model_clffolder(folders[3], df_subfp, failed_ls=failed_subfp, scale=False, validnum=5)
    te_pred[:, 20:25] = model_clffolder(folders[4], df_padel2d, failed_ls=failed_padel2d, scale=True, validnum=5)
    te_pred[:, 25:30] = model_clffolder(folders[5], df_rdkit2d, failed_ls=failed_rdkit2d, scale=True, validnum=5)

    score, conf = np.zeros(shape=te_pred.shape[0]), np.zeros(shape=te_pred.shape[0])
    for i in range(te_pred.shape[0]):
        score[i], conf[i] = cal_score_conf(te_pred[i], tr_avgY)

    # result
    result_dict = {
        'Caco2_Perm10_clf_V5beta3_Score': score,
        'Caco2_Perm10_clf_V5beta3_Conf': conf,
    }
    
    df_result = pd.DataFrame(result_dict)
    print('Caco2_Perm10_clf_V5beta3_Score, done.')
    
    return df_result


def extract_id(file):
    with open(file) as f:
        failed_ls = [int(line.strip()) for line in f]
    return failed_ls



def summary_score(df, target_title, score_title, out_txt):

    N_posi = df.loc[df.loc[:, target_title] == 1].shape[0]
    N_nega = df.loc[df.loc[:, target_title] == 0].shape[0]
    print('Num of posi {}, num of nega {}'.format(N_posi, N_nega))

    range_left  = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                   0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    range_right = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    

    with open(out_txt, 'w') as f:
        f.write('N_posi = {}, N_nega = {}\n\n'.format(N_posi, N_nega))

        f.write('Range\tn_posi\tpercent_posi\tn_nega\tpercent_nega\tratio\n')
    

    for l, r in zip(range_left, range_right):

        sel = df.loc[:, score_title].values
        sel = np.where( (sel >= l) & (sel < r), True, False)

        df_sel = df.loc[sel].reset_index(drop=True)

        n_posi = df_sel.loc[df_sel.loc[:, target_title] == 1].shape[0]
        n_nega = df_sel.loc[df_sel.loc[:, target_title] == 0].shape[0]

        percent_posi = 100 * n_posi / N_posi
        percent_nega = 100 * n_nega / N_nega

        try: ratio = percent_posi / (percent_posi + percent_nega)
        except ZeroDivisionError: ratio = -1.0

        with open(out_txt, 'a') as f:

            if r != 1.01:
                f.write('[{}, {})\t{}\t{:.2f}%\t{}\t{:.2f}%\t{:.3f}\n'\
                    .format(l, r, n_posi, percent_posi, n_nega, percent_nega, ratio))
            else:
                f.write('[{}, 1.0]\t{}\t{:.2f}%\t{}\t{:.2f}%\t{:.3f}\n'\
                    .format(l, n_posi, percent_posi, n_nega, percent_nega, ratio))


def main(curated_file, prod_folder, trte, target):

    modeling_folder = os.path.dirname(curated_file) + '/'

    # read modeldf files
    df_maccs     = pd.read_csv(modeling_folder + 'GroupA_MACCS_modeldf.csv')
    df_ecfp4     = pd.read_csv(modeling_folder + 'GroupB_ECFP4-1024_modeldf.csv')
    df_pubchemfp = pd.read_csv(modeling_folder + 'GroupH_PubChemFP-881_modeldf.csv')
    df_subfp     = pd.read_csv(modeling_folder + 'GroupL_SubFP-307_modeldf.csv')
    df_padel2d   = pd.read_csv(modeling_folder + 'GroupP_PaDEL2D-784_modeldf.csv')
    df_rdkit2d   = pd.read_csv(modeling_folder + 'GroupR_RDKit2D-208_modeldf.csv')

    # read failed id
    f_pubchemfp = modeling_folder + 'GroupH_PubChemFP-881_failedID.txt'
    f_subfp     = modeling_folder + 'GroupL_SubFP-307_failedID.txt'
    f_padel2d   = modeling_folder + 'GroupP_PaDEL2D-784_failedID.txt'
    f_rdkit2d   = modeling_folder + 'GroupR_RDKit2D-208_failedID.txt'

    if os.path.exists(f_pubchemfp): failed_pubchemfp = extract_id(f_pubchemfp)
    else: failed_pubchemfp = []

    if os.path.exists(f_subfp): failed_subfp = extract_id(f_subfp)
    else: failed_subfp = []

    if os.path.exists(f_padel2d): failed_padel2d = extract_id(f_padel2d)
    else: failed_padel2d = []

    if os.path.exists(f_rdkit2d): failed_rdkit2d = extract_id(f_rdkit2d)
    else: failed_rdkit2d = []

    # read curated file
    df_curated = pd.read_csv(curated_file)

    # prepare result DataFrame
    df_out = df_curated.loc[:, ['NO', trte, target]]

    ################## predict, need modify, START ##################################
    df_endpoint = clf_prod_ABHLPR(df_maccs, df_ecfp4, df_pubchemfp, failed_pubchemfp, df_subfp, failed_subfp, df_padel2d, failed_padel2d, df_rdkit2d, failed_rdkit2d)
    # df_endpoint = clf_HSF1_V1beta3(df_maccs, df_ecfp4, df_pubchemfp, failed_pubchemfp, df_padel2d, failed_padel2d, df_rdkit2d, failed_rdkit2d)
    ################## predict, need modify, END ##################################

    df_endpoint.columns = ['Score', 'Conf']
    df_out = pd.concat([df_out, df_endpoint], axis=1)
    df_out.to_csv(prod_folder + 'Score.csv', index=None)

    # cal metrics
    trte_ar = df_out.loc[:, trte].values

    tr_Y = df_out.loc[trte_ar == 'train', target].values
    te_Y = df_out.loc[trte_ar == 'test', target].values
    tr_avgY = round(tr_Y.mean(), 3)
    print('train avgY = {}'.format(tr_avgY))

    tr_score = df_out.loc[trte_ar == 'train', 'Score'].values
    te_score = df_out.loc[trte_ar == 'test', 'Score'].values

    tr_preY, te_preY = np.zeros(shape=tr_Y.shape[0]), np.zeros(shape=te_Y.shape[0])
    tr_preY[tr_score >= tr_avgY] = 1
    te_preY[te_score >= tr_avgY] = 1

    tr_auc = roc_auc_score(tr_Y, tr_score)
    te_auc = roc_auc_score(te_Y, te_score)

    tr_tp, tr_tn, tr_fp, tr_fn, tr_se, tr_sp, tr_acc, tr_mcc = Metrics(tr_Y, tr_preY)
    te_tp, te_tn, te_fp, te_fn, te_se, te_sp, te_acc, te_mcc = Metrics(te_Y, te_preY)

    tr_fpr, tr_tpr, tr_thresholds = roc_curve(tr_Y, tr_score)
    te_fpr, te_tpr, te_thresholds = roc_curve(te_Y, te_score)

    tr_conf = df_out.loc[trte_ar == 'train', 'Conf'].values
    te_conf = df_out.loc[trte_ar == 'test', 'Conf'].values

    print("training set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(tr_tp, tr_tn, tr_fp, tr_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(tr_se, tr_sp, tr_acc, tr_mcc, tr_auc))

    print("test set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(te_tp, te_tn, te_fp, te_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(te_se, te_sp, te_acc, te_mcc, te_auc))

    # write logfile
    logfile = prod_folder + 'metrics.log'

    log_string = ''
    log_string += 'train\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(tr_tp, tr_tn, tr_fp, tr_fn, tr_se*100, tr_sp*100, tr_acc*100, tr_mcc, tr_auc)
    log_string += 'test\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(te_tp, te_tn, te_fp, te_fn, te_se*100, te_sp*100, te_acc*100, te_mcc, te_auc)
    
    with open(logfile, 'w') as f:
        f.write(log_string)


    # plotting
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.4 * 3, 4.8 * 2))

    tr_posi_weight = [1.0 / tr_Y[tr_Y == 1].shape[0]] * tr_Y[tr_Y == 1].shape[0]
    tr_nega_weight = [1.0 / tr_Y[tr_Y == 0].shape[0]] * tr_Y[tr_Y == 0].shape[0]

    ax[0][0].hist([tr_score[tr_Y == 0], tr_score[tr_Y == 1]],
                   weights=[tr_nega_weight, tr_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[0][0].legend(loc='upper right')
    ax[0][0].set_xlabel('modeling score')
    ax[0][0].set_ylabel('number of compounds (%)')
    ax[0][0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[0][0].set_title('training set score')

    te_posi_weight = [1.0 / te_Y[te_Y == 1].shape[0]] * te_Y[te_Y == 1].shape[0]
    te_nega_weight = [1.0 / te_Y[te_Y == 0].shape[0]] * te_Y[te_Y == 0].shape[0]

    ax[0][1].hist([te_score[te_Y == 0], te_score[te_Y == 1]],
                   weights=[te_nega_weight, te_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[0][1].legend(loc='upper right')
    ax[0][1].set_xlabel('modeling score')
    ax[0][1].set_ylabel('number of compounds')
    ax[0][1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[0][1].set_title('test set score')

    ## ROC Curve
    ax[0][2].plot([0, 1], [0, 1], 'k')
    ax[0][2].plot(tr_fpr, tr_tpr, 'k', label='training, AUC={:.3f}'.format(tr_auc))
    ax[0][2].plot(te_fpr, te_tpr, 'r--', label='test, AUC={:.3f}'.format(te_auc))
    ax[0][2].set_xlabel('FPR')
    ax[0][2].set_ylabel('TPR')
    ax[0][2].set_title('ROC curve')
    ax[0][2].legend(loc='lower right')

    ## Confidence
    # ax[1][0].scatter(tr_conf[tr_Y == 0], tr_score[tr_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    # ax[1][0].scatter(tr_conf[tr_Y == 1], tr_score[tr_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][0].scatter(tr_conf[np.where((tr_Y == 0)&(tr_score < tr_avgY))], tr_score[np.where((tr_Y == 0)&(tr_score < tr_avgY))],
                     marker='x', color='b', alpha=0.7, label='negative')
    ax[1][0].scatter(tr_conf[np.where((tr_Y == 1)&(tr_score >= tr_avgY))], tr_score[np.where((tr_Y == 1)&(tr_score >= tr_avgY))],
                     marker='x', color='r', alpha=0.8, label='positive')
    ax[1][0].scatter(tr_conf[np.where((tr_Y == 1)&(tr_score < tr_avgY))], tr_score[np.where((tr_Y == 1)&(tr_score < tr_avgY))],
                     marker='x', color='r', alpha=0.8)
    ax[1][0].scatter(tr_conf[np.where((tr_Y == 0)&(tr_score >= tr_avgY))], tr_score[np.where((tr_Y == 0)&(tr_score >= tr_avgY))],
                     marker='x', color='b', alpha=0.7)

    ax[1][0].set_ylim(-0.03, 1.03)
    ax[1][0].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][0].set_ylabel('Score')
    ax[1][0].set_title('Training confidence: train avgY = {:.3f}'.format(tr_avgY))
    ax[1][0].legend(loc='best')

    # ax[1][1].scatter(te_conf[te_Y == 0], te_score[te_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    # ax[1][1].scatter(te_conf[te_Y == 1], te_score[te_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][1].scatter(te_conf[np.where((te_Y == 0)&(te_score < tr_avgY))], te_score[np.where((te_Y == 0)&(te_score < tr_avgY))],
                     marker='x', color='b', alpha=0.7, label='negative')
    ax[1][1].scatter(te_conf[np.where((te_Y == 1)&(te_score >= tr_avgY))], te_score[np.where((te_Y == 1)&(te_score >= tr_avgY))],
                     marker='x', color='r', alpha=0.8, label='positive')
    ax[1][1].scatter(te_conf[np.where((te_Y == 1)&(te_score < tr_avgY))], te_score[np.where((te_Y == 1)&(te_score < tr_avgY))],
                     marker='x', color='r', alpha=0.8)
    ax[1][1].scatter(te_conf[np.where((te_Y == 0)&(te_score >= tr_avgY))], te_score[np.where((te_Y == 0)&(te_score >= tr_avgY))],
                     marker='x', color='b', alpha=0.7)
    ax[1][1].set_ylim(-0.03, 1.03)
    ax[1][1].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][1].set_ylabel('Score')
    ax[1][1].set_title('Test confidence')
    ax[1][1].legend(loc='best')

    ax[1][2].set_axis_off()
    
    plt.savefig(prod_folder + 'build.png', dpi=600, bbox_inches='tight')


    # summary_score
    summary_score(df_out, target, 'Score', prod_folder + 'summary_score.txt')

    print('Done.')



if __name__ == '__main__':

    main(
        curated_file = '/home/cadd/Project/Permeability_Caco2/modelV5beta3_Perm10_RFC/datasetV5beta3_Caco2_Perm10_curated826_trtecv.csv',
        prod_folder = '/home/cadd/Project/Permeability_Caco2/Caco2_Perm10_modelV5beta3_prodRFC_ABHLPR/',
        trte = 'trte',
        target = 'target_binary10',
    )
