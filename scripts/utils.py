import math
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBA, CalcNumHBD, CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolLogP
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as image
from rdkit.Chem.MolStandardize import rdMolStandardize
import json

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

def save_sdf(df, smi_title, sdf_file):
    if os.path.exists(sdf_file):
        print('SDF File exists, it will be overwriten.')
        os.remove(sdf_file)
    
    smiles_ar = df.loc[:, smi_title].values
    print_every = 100000
    
    temp_string = ''
    
    for i, smi in enumerate(smiles_ar):
        molblock = Chem.MolToMolBlock(Chem.MolFromSmiles(smi))
        temp_string += molblock + '$$$$\n'
        
        if (i+1) % print_every == 0:
            with open(sdf_file, 'a') as f:
                f.write(temp_string)
            temp_string = ''
            print('  save SDF, processing {} / {} compds, ...'.format(i+1, smiles_ar.shape[0]))
    
    with open(sdf_file, 'a') as f:
        f.write(temp_string)
    print('  save SDF, processing {} / {} compds, done.'.format(i+1, smiles_ar.shape[0]))

def generate_tautomer(df, smiles_title, tautomer_smiles_prefix='tautomer', print_every=100):
    time_start = time.time()
    
    smiles_ar = df.loc[:, smiles_title].values
    
    df_out = pd.DataFrame()
    
    for i, smi in enumerate(smiles_ar):
        tautomer_generator = rdMolStandardize.TautomerEnumerator()
        result = tautomer_generator.Enumerate(Chem.MolFromSmiles(smi))
        
        _smi = list(result.smilesTautomerMap.keys())
        _score = [int(tautomer_generator.ScoreTautomer(Chem.MolFromSmiles(s))) for s in _smi]
        
        _maxscore = [0] * len(_score)
        for j, score in enumerate(_score):
            if score == np.max(_score):
                _maxscore[j] = 1
        
        _rankscore = [0] * len(_score)
        rank = len(_score)
        for j in np.argsort(_score):
            _rankscore[j] = rank
            rank -= 1
        
        _same = [0] * len(_smi)
        try:
            inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
            _inchikey = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in _smi]
            
            for j, inchi in enumerate(_inchikey):
                if inchi == inchikey:
                    _same[j] = 1
        except:
            pass
        
        _dict = {
            tautomer_smiles_prefix + ' SMILES': _smi,
            tautomer_smiles_prefix + ' Score': _score,
            tautomer_smiles_prefix + ' Max Score': _maxscore,
            tautomer_smiles_prefix + ' Rank Score': _rankscore,
            tautomer_smiles_prefix + ' Same': _same
        }
        
        _df = pd.concat([df.loc[[i] * len(_smi)].reset_index(drop=True),
                         pd.DataFrame(_dict)], axis=1)
        
        df_out = pd.concat([df_out, _df], axis=0).reset_index(drop=True)
        
        if (i + 1) % print_every == 0:
            print('Process {} / {}, time {}'.format(i+1, smiles_ar.shape[0], cal_time(time_start)))
        
    print('Process {} / {}, time {}'.format(i+1, smiles_ar.shape[0], cal_time(time_start)))
    print('Result: {} tautomers have been generated.'.format(df_out.shape[0]))
    
    return df_out


def _cal_distance(center, X):
    dist_ar = np.zeros(shape=X.shape[0])
    
    for i in range(X.shape[0]):
        dist = np.sqrt(np.sum((center - X[i]) ** 2))
        dist_ar[i] = dist
    return dist_ar

def trtesplit_by_TSNE(df, smiles_title, trte_title='trte', target_title='target', STEP=5):
    '''Example:
    >>> result_df, result_dict = trtesplit_by_TSNE(**kward)
    '''
    smiles_ar = df.loc[:, smiles_title].values
    targets = df.loc[:, target_title].values

    des_ar = np.zeros(shape=(smiles_ar.shape[0], 6))
    
    for i, smi in enumerate(smiles_ar):
        mol = Chem.MolFromSmiles(smi)
        des_ar[i, 0] = CalcExactMolWt(mol)
        des_ar[i, 1] = CalcNumHBA(mol)
        des_ar[i, 2] = CalcNumHBD(mol)
        des_ar[i, 3] = CalcTPSA(mol)
        des_ar[i, 4] = CalcNumRotatableBonds(mol)
        des_ar[i, 5] = MolLogP(mol)
    
    scaler = StandardScaler()
    des_ar = scaler.fit_transform(des_ar)
    
    tsne = TSNE(n_iter=10000, init='pca', random_state=42)
    tsne_ar = tsne.fit_transform(des_ar)
    
    scaler = MinMaxScaler()
    tsne_ar = scaler.fit_transform(tsne_ar)
    
    max_score = 0

    for n_cluster in range(2, 6):
        _kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        _cluster_labels = _kmeans.fit_predict(tsne_ar)
        _score = silhouette_score(tsne_ar, _cluster_labels)
        print('n_cluster = {}, Score = {:.3f}'.format(n_cluster, _score))

        if _score > max_score:
            max_score = _score
            best_cluster = n_cluster
            labels = _cluster_labels
            kmeans = _kmeans
    
    centers = kmeans.cluster_centers_
    # calculate the distance between center and samples
    result_df = pd.DataFrame()
    
    for cluster in range(best_cluster):
        for target in set(targets):
            _id = np.where((labels == cluster) & (targets == target))[0]
            _dist = _cal_distance(centers[cluster], tsne_ar[_id])
            _df = pd.DataFrame({'ID': _id, 'DIST': _dist})
            _df = _df.sort_values(by='DIST', ascending=False).reset_index(drop=True)

            # if _id.shape[0] >= 5:
            if _id.shape[0] >= STEP:
                _trte = np.zeros(shape=_id.shape[0])
                # for i in range(4, _id.shape[0], 5):
                for i in range(STEP - 1, _id.shape[0], STEP):
                    _trte[i] = 1
                # _sr = pd.Series(_trte, name=trte_title).replace(1, 'test').replace(0, 'train')
                _sr = pd.Series(_trte, name=trte_title).replace([1, 0], ['test', 'train'])
            else:
                _sr = pd.Series(['train'] * _id.shape[0], name=trte_title)
            
            _df = pd.concat([_df, _sr], axis=1)
            result_df = pd.concat([result_df, _df]).reset_index(drop=True)
    
    result_df = result_df.sort_values(by='ID', ascending=True).reset_index(drop=True)
    result_df = pd.concat([result_df,
                           pd.Series(labels, name='cluster_label'),
                           pd.Series(targets, name=target_title)], axis=1)

    return_df = pd.concat([df, result_df.loc[:, trte_title]], axis=1)

    return_dict = {
        'df': result_df,
        'tsne_ar': tsne_ar,
        'centers': centers
    }

    return return_df, return_dict


### RDKit Drawing Utils
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(0, atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx() + 1))
    return mol

def rdMolToImageArray(mol, size=400):
    d = Draw.rdMolDraw2D.MolDraw2DSVG(size, size)
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    with open('__temp__.svg', 'w') as f:
        f.write(svg)
    
    cairosvg.svg2png(url='__temp__.svg',
                     dpi=1000,
                     scale=2,
                     write_to='__temp__.png')
    
    img = image.imread('__temp__.png')
    
    os.remove('__temp__.svg')
    os.remove('__temp__.png')
    
    return img


def merge_dict(*dn):
    merge = {}
    for d in dn:
        merge = {**merge, **d}
    return merge

def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))

def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d


if __name__ == '__main__':
    pass
