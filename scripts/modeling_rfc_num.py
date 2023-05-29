import os
import time
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.ticker import PercentFormatter

from utils_public import load_json, cal_time


def cal_conf(ar, avgY):
    mean = np.mean(ar)
    std  = np.std(ar)

    if std == 0:
        conf = 0.0
    else:
        conf = min(
            scipy.stats.norm(mean, std).cdf(avgY),
            1 - scipy.stats.norm(mean, std).cdf(avgY)
        )
    
    return conf


def extract_XY(df, trtecv_title, valid_title, des_title, target, scaler):
    tr_X = scaler.transform(df.loc[df.loc[:, trtecv_title] != valid_title, des_title].values)
    tr_Y = df.loc[df.loc[:, trtecv_title] != valid_title, target].values
    va_X = scaler.transform(df.loc[df.loc[:, trtecv_title] == valid_title, des_title].values)
    va_Y = df.loc[df.loc[:, trtecv_title] == valid_title, target].values

    return tr_X, tr_Y, va_X, va_Y

def grid_search(tr_X, tr_Y, va_X, va_Y):

    # 4 x 2 x 4 x 8 = 256
    params = {'n_estimators': [50, 100, 200, 400],
              'criterion':    ['entropy', 'gini'],
              'max_features': ['log2', 0.25, 0.5, 1.0],
              'max_depth':    [i for i in range(1, 9, 1)]}

    process_all = 1
    for key in params.keys():
        process_all *= len(params[key])
    
    best_tr = -1
    best_va = -1
    best_overfit = np.Infinity

    warnings.filterwarnings('ignore')

    process = 0
    print_every = 64

    time_start = time.time()

    dict_grid = {
        'params': [],
        'tr_mcc': [],
        'va_mcc': [],
        'overfit': [],
    }

    for n_estimators in params['n_estimators']:
        for criterion in params['criterion']:
            for max_features in params['max_features']:
                for max_depth in params['max_depth']:
                    
                    process += 1

                    dict_grid['params'].append([n_estimators, criterion, max_features, max_depth])

                    clf = RandomForestClassifier(n_estimators = n_estimators,
                                                 criterion    = criterion,
                                                 max_features = max_features,
                                                 max_depth    = max_depth,
                                                 random_state = 42)
                    clf.fit(tr_X, tr_Y)

                    avgY = tr_Y.mean()
                    tr_score = clf.predict_proba(tr_X)[:, 1]
                    va_score = clf.predict_proba(va_X)[:, 1]

                    tr_preY = np.zeros(shape=tr_Y.shape[0])
                    va_preY = np.zeros(shape=va_Y.shape[0])

                    tr_preY[tr_score >= avgY] = 1
                    va_preY[va_score >= avgY] = 1
                    
                    tr_mcc = round(matthews_corrcoef(tr_Y, tr_preY), 3)
                    va_mcc = round(matthews_corrcoef(va_Y, va_preY), 3)
                    
                    if tr_mcc >= 0.99:
                        if va_mcc >= 0.98:
                            overfit = 1.0
                        else:
                            overfit = 100.0
                    else:
                        overfit = (1 - va_mcc) / (1 - tr_mcc)

                    if va_mcc < best_va:
                        pass
                    else:
                        if tr_mcc <= 0.7:
                            best_tr = tr_mcc
                            best_va = va_mcc
                            best_overfit = overfit
                            best_params = [n_estimators, criterion, max_features, max_depth]
                        else:
                            if overfit < max(1.6, best_overfit):
                                best_tr = tr_mcc
                                best_va = va_mcc
                                best_overfit = overfit
                                best_params = [n_estimators, criterion, max_features, max_depth]
                            else:
                                pass
                    
                    dict_grid['tr_mcc'].append(tr_mcc)
                    dict_grid['va_mcc'].append(va_mcc)
                    dict_grid['overfit'].append(overfit)
                    
                    if process % print_every == 0:
                        print('  grid search, processing {} / {} ({}) ...'.format(process, process_all, cal_time(time_start)))
    
    print('  best tr_mcc = {:.3f}, va_mcc = {:.3f}, overfit = {:.3f}'.format(best_tr, best_va, best_overfit))
    print('  best params: n_estimators, criterion, max_features, max_depth = {}, {}, {}, {}'\
          .format(best_params[0], best_params[1], best_params[2], best_params[3]))
    
    return best_params, best_tr, best_va, best_overfit, dict_grid


def Metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try: se = float(tp) / float(tp + fn)
    except ZeroDivisionError: se = -1.0
    try: sp = float(tn) / float(tn + fp)
    except ZeroDivisionError: sp = -1.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return tp, tn, fp, fn, se, sp, acc, mcc


def build_rfc_num_model(modeldf_file, json_file, trte_title, trtecv_title, target, des, prefix, validnum=5):

    print('Start calculation ...\n')
    time_start = time.time()

    des_dict = load_json(json_file)

    if des in des_dict.keys():
        model_des = des_dict[des]
    else:
        print('{} not in des_dict.'.format(des))
        return None

    folder = os.path.dirname(modeldf_file) + '/{}_{}_RFC/'.format(prefix, des)
    logfile = folder + 'metrics.log'
    log_string = ''

    if os.path.exists(folder):
        pass
    else:
        os.makedirs(folder)
        print('Make new folder: {}\n'.format(folder))
    
    model_df = pd.read_csv(modeldf_file)

    tr_df = model_df.loc[model_df.loc[:, trte_title] == 'train'].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(tr_df.loc[:, model_des].values)
    joblib.dump(scaler, folder + 'build.scaler')

    clfs = []
    cv_Y = []
    cv_preY = []
    cv_score = []

    for valid in ['train' + str(i) for i in range(1, validnum+1)]:
        print('processing: {}, start ...'.format(valid))

        tr_X, tr_Y, va_X, va_Y = extract_XY(tr_df, trtecv_title, valid, model_des, target, scaler)
        best_params, best_tr, best_va, best_overfit, dict_grid = grid_search(tr_X, tr_Y, va_X, va_Y)

        clf = RandomForestClassifier(n_estimators = best_params[0],
                                     criterion    = best_params[1],
                                     max_features = best_params[2],
                                     max_depth    = best_params[3],
                                     random_state = 42)
        clf.fit(tr_X, tr_Y)

        tr_avgY = tr_Y.mean()
        va_score = clf.predict_proba(va_X)[:, 1]
        va_preY = np.zeros(shape=va_Y.shape[0])
        va_preY[va_score >= tr_avgY] = 1

        clfs.append(clf)
        cv_Y = np.append(cv_Y, va_Y)
        cv_preY = np.append(cv_preY, va_preY)
        cv_score = np.append(cv_score, va_score)

        joblib.dump(clf, folder + valid + '.model')

        log_string += '{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n'\
            .format(valid, best_params[0], best_params[1], best_params[2], best_params[3],
                    best_tr, best_va, best_overfit)
        
        # save dict_grid
        df_grid = pd.DataFrame(dict_grid)
        df_grid = df_grid.sort_values(by='va_mcc', ascending=False).reset_index(drop=True)
        df_grid.to_csv(folder + '{}_grid.txt'.format(valid), sep='\t', index=None)

        print('processing: {} done, time {}\n'.format(valid, cal_time(time_start)))

    cv_auc = roc_auc_score(cv_Y, cv_score)
    cv_fpr, cv_tpr, cv_thresholds = roc_curve(cv_Y, cv_score)
    cv_tp, cv_tn, cv_fp, cv_fn, cv_se, cv_sp, cv_acc, cv_mcc = Metrics(cv_Y, cv_preY)
    
    tr_X = scaler.transform(model_df.loc[model_df.loc[:, trte_title] == 'train', model_des].values)
    tr_Y = model_df.loc[model_df.loc[:, trte_title] == 'train', target].values
    te_X = scaler.transform(model_df.loc[model_df.loc[:, trte_title] == 'test', model_des].values)
    te_Y = model_df.loc[model_df.loc[:, trte_title] == 'test', target].values
    
    tr_pred = np.zeros(shape=(tr_Y.shape[0], 5))
    te_pred = np.zeros(shape=(te_Y.shape[0], 5))
    
    for i, clf in enumerate(clfs):
        tr_pred[:, i] = clf.predict_proba(tr_X)[:, 1]
        te_pred[:, i] = clf.predict_proba(te_X)[:, 1]
    
    tr_score = tr_pred.mean(axis=1)
    te_score = te_pred.mean(axis=1)

    tr_avgY = round(tr_Y.mean(), 3)
    print('train avgY = {}'.format(tr_avgY))

    tr_preY = np.zeros(shape=tr_Y.shape[0])
    te_preY = np.zeros(shape=te_Y.shape[0])
    tr_preY[tr_score >= tr_avgY] = 1
    te_preY[te_score >= tr_avgY] = 1

    tr_auc = roc_auc_score(tr_Y, tr_score)
    te_auc = roc_auc_score(te_Y, te_score)

    tr_tp, tr_tn, tr_fp, tr_fn, tr_se, tr_sp, tr_acc, tr_mcc = Metrics(tr_Y, tr_preY)
    te_tp, te_tn, te_fp, te_fn, te_se, te_sp, te_acc, te_mcc = Metrics(te_Y, te_preY)

    tr_fpr, tr_tpr, tr_thresholds = roc_curve(tr_Y, tr_score)
    te_fpr, te_tpr, te_thresholds = roc_curve(te_Y, te_score)

    tr_conf = np.array([cal_conf(tr_pred[i], tr_avgY) for i in range(tr_pred.shape[0])])
    te_conf = np.array([cal_conf(te_pred[i], tr_avgY) for i in range(te_pred.shape[0])])

    pd.concat([
        pd.Series(tr_Y, name='tr_Y'),
        pd.DataFrame(np.around(tr_pred, 3), columns=['valid{}'.format(i) for i in range(1, validnum+1)]),
        pd.Series(np.around(tr_score, 3), name='tr_score'),
        pd.Series(np.around(tr_conf, 3), name='tr_conf')
    ], axis=1).to_csv(folder + 'tr_result.csv')

    pd.concat([
        pd.Series(te_Y, name='te_Y'),
        pd.DataFrame(np.around(te_pred, 3), columns=['valid{}'.format(i) for i in range(1, validnum+1)]),
        pd.Series(np.around(te_score, 3), name='te_score'),
        pd.Series(np.around(te_conf, 3), name='te_conf')
    ], axis=1).to_csv(folder + 'te_result.csv')

    with open(folder + 'model_des.txt', 'w') as f:
        for s in model_des:
            f.write(s + '\n')

    print("training set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(tr_tp, tr_tn, tr_fp, tr_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(tr_se, tr_sp, tr_acc, tr_mcc, tr_auc))

    print("validation set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(cv_tp, cv_tn, cv_fp, cv_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(cv_se, cv_sp, cv_acc, cv_mcc, cv_auc))

    print("test set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(te_tp, te_tn, te_fp, te_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(te_se, te_sp, te_acc, te_mcc, te_auc))
    
    
    log_string += 'train\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(tr_tp, tr_tn, tr_fp, tr_fn, tr_se*100, tr_sp*100, tr_acc*100, tr_mcc, tr_auc)

    log_string += 'valid\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(cv_tp, cv_tn, cv_fp, cv_fn, cv_se*100, cv_sp*100, cv_acc*100, cv_mcc, cv_auc)
    
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
    ax[0][2].plot(cv_fpr, cv_tpr, 'b--', label='validation, AUC={:.3f}'.format(cv_auc))
    ax[0][2].plot(te_fpr, te_tpr, 'r--', label='test, AUC={:.3f}'.format(te_auc))
    ax[0][2].set_xlabel('FPR')
    ax[0][2].set_ylabel('TPR')
    ax[0][2].set_title('ROC curve')
    ax[0][2].legend(loc='lower right')

    ## Confidence
    ax[1][0].scatter(tr_conf[tr_Y == 0], tr_score[tr_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    ax[1][0].scatter(tr_conf[tr_Y == 1], tr_score[tr_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][0].set_ylim(-0.03, 1.03)
    ax[1][0].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][0].set_ylabel('Score')
    ax[1][0].set_title('Training confidence: train avgY = {:.3f}'.format(tr_avgY))
    ax[1][0].legend(loc='best')

    ax[1][1].scatter(te_conf[te_Y == 0], te_score[te_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    ax[1][1].scatter(te_conf[te_Y == 1], te_score[te_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][1].set_ylim(-0.03, 1.03)
    ax[1][1].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][1].set_ylabel('Score')
    ax[1][1].set_title('Test confidence')
    ax[1][1].legend(loc='best')


    ## feature importance
    feature_importance = np.zeros(shape=len(model_des))

    for clf in clfs:
        feature_importance += clf.feature_importances_
    
    feature_importance = feature_importance / len(clfs)

    sort_importance = -np.sort(-feature_importance)
    sort_importance_index = np.argsort(-feature_importance)
    
    if feature_importance.shape[0] > 10:
        show_num = 10
    else:
        show_num = feature_importance.shape[0]
    
    ax[1][2].bar(range(show_num),
                 sort_importance[:show_num],
                 tick_label=[model_des[i] for i in sort_importance_index[:show_num]])
    
    for tick in ax[1][2].get_xticklabels():
        tick.set_rotation(30)
    
    plt.savefig(folder + 'build.png', dpi=600, bbox_inches='tight')

    print('Calculation done. Time {}\n'.format(cal_time(time_start)))

