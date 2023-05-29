import re
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def CheckSMI(smi):
    
    # step 1
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None: pass
        else: return False
    except:
        return False

    # step 2
    remover = rdMolStandardize.FragmentRemover()
    try:
        mol = remover.remove(mol)
        mol = rdMolStandardize.FragmentParent(mol)
    except:
        return False
    
    # step 3
    try: inchikey = Chem.MolToInchiKey(mol)
    except:return False

    return True

def StandardFrag(smi, mode='frag'):
    ''' including: 
        1) Kekulize smi
        2) mode = 'frag': transfer '[1*]' to '*'
        3) mode = 'core': transfer '[X*]' and '[X*] ...' to '[1*] and [2*] ...'
        4) mode = 'out': transfer '[1*]' to '[*]'
    '''
    m = Chem.MolFromSmiles(smi)
    Chem.Kekulize(m, clearAromaticFlags=True)
    s = Chem.MolToSmiles(m)

    regex = '(\[[^\[\]]{1,6}\])'
    split_ls = re.split(regex, s)
    dummy = [x for x in split_ls if '*' in x]
    
    if mode == 'frag':
        for x in dummy: s = s.replace(x, '*', 1)
    elif mode == 'core':
        dummy_renum = ['[{}*]'.format(i) for i in range(1, len(dummy)+1)]
        for old, new in zip(dummy, dummy_renum):
            s = s.replace(old, new, 1)
    elif mode == 'out':
        for x in dummy: s = s.replace(x, '[*]', 1)
    else:
        return None

    return s

def MolGenForSingle(main_smi, frags_ls):
    
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # main_smi and fraglib (remove BRICS dummy, transfer to '*' only)
    fraglib = [StandardFrag(x, mode='frag') for x in frags_ls] # fraglib and frags_ls have same length and order
    main_smi = '[1*]' + StandardFrag(main_smi, mode='frag')[1:]

    # generate
    gen_smiles = []     # fraglib and gen_smiles have same length and order

    for frag in fraglib:
        mol = AllChem.ReplaceSubstructs(Chem.MolFromSmiles(main_smi),
                                        Chem.MolFromSmiles('[1*]'),
                                        Chem.MolFromSmiles(frag[1:]))[0]
        gen_smiles.append(Chem.MolToSmiles(mol))
    
    # Check SMILES
    success_smiles, success_frags = [], []
    for smi, frag in zip(gen_smiles, frags_ls):
        if CheckSMI(smi):
            success_smiles.append(smi)
            success_frags.append(frag)

    return success_frags, success_smiles

def ReplaceDouble(core_smi, frag1_smi, frag2_smi):
    
    mol = AllChem.ReplaceSubstructs(Chem.MolFromSmiles(core_smi),
                                    Chem.MolFromSmiles('[1*]'),
                                    Chem.MolFromSmiles(frag1_smi))[0]

    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[2*]'),
                                    Chem.MolFromSmiles(frag2_smi))[0]
    
    return Chem.MolToSmiles(mol)
 

def MolGenForDouble(frags_string, cores_ls):
    
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # split frags_string to two frags
    frags = frags_string.split('.')
    frag1_smi, frag2_smi = [StandardFrag(frags[0], mode='frag')[1:], StandardFrag(frags[1], mode='frag')[1:]]

    # corelib (transfer [X*] and [Y*] to [1*] and [2*])
    corelib = [StandardFrag(x, mode='core') for x in cores_ls]

    # generate
    gen_smiles = []
    used_cores = []
    
    for core, core_BRICS_dummy in zip(corelib, cores_ls):
        gen_smiles.append(ReplaceDouble(core, frag1_smi, frag2_smi))
        gen_smiles.append(ReplaceDouble(core, frag2_smi, frag1_smi))
        used_cores += [core_BRICS_dummy] * 2
        
    # Check smiles
    success_smiles, success_cores = [], []
    for smi, core in zip(gen_smiles, used_cores):
        if CheckSMI(smi):
            success_smiles.append(smi)
            success_cores.append(core)

    return success_cores, success_smiles 

def ReplaceTriple(core_smi, frag1_smi, frag2_smi, frag3_smi):
    
    mol = AllChem.ReplaceSubstructs(Chem.MolFromSmiles(core_smi),
                                    Chem.MolFromSmiles('[1*]'),
                                    Chem.MolFromSmiles(frag1_smi))[0]

    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[2*]'),
                                    Chem.MolFromSmiles(frag2_smi))[0]
    
    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[3*]'),
                                    Chem.MolFromSmiles(frag3_smi))[0]
    
    return Chem.MolToSmiles(mol)
 

def MolGenForTriple(frags_string, cores_ls):
    
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # split frags_string to three frags
    frags = frags_string.split('.')
    frag1_smi = StandardFrag(frags[0], mode='frag')[1:]
    frag2_smi = StandardFrag(frags[1], mode='frag')[1:]
    frag3_smi = StandardFrag(frags[2], mode='frag')[1:]

    # corelib
    corelib = [StandardFrag(x, mode='core') for x in cores_ls]

    # generate
    gen_smiles = []
    used_cores = []
    
    for core, core_BRICS_dummy in zip(corelib, cores_ls):
        gen_smiles.append(ReplaceTriple(core, frag1_smi, frag2_smi, frag3_smi))
        gen_smiles.append(ReplaceTriple(core, frag1_smi, frag3_smi, frag2_smi))
        gen_smiles.append(ReplaceTriple(core, frag2_smi, frag1_smi, frag3_smi))
        gen_smiles.append(ReplaceTriple(core, frag2_smi, frag3_smi, frag1_smi))
        gen_smiles.append(ReplaceTriple(core, frag3_smi, frag1_smi, frag2_smi))
        gen_smiles.append(ReplaceTriple(core, frag3_smi, frag2_smi, frag1_smi))
        used_cores += [core_BRICS_dummy] * 6
        
    # Check smiles
    success_smiles, success_cores = [], []
    for smi, core in zip(gen_smiles, used_cores):
        if CheckSMI(smi):
            success_smiles.append(smi)
            success_cores.append(core)

    return success_cores, success_smiles

def ReplaceTetra(core_smi, frag1_smi, frag2_smi, frag3_smi, frag4_smi):
    
    mol = AllChem.ReplaceSubstructs(Chem.MolFromSmiles(core_smi),
                                    Chem.MolFromSmiles('[1*]'),
                                    Chem.MolFromSmiles(frag1_smi))[0]

    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[2*]'),
                                    Chem.MolFromSmiles(frag2_smi))[0]
    
    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[3*]'),
                                    Chem.MolFromSmiles(frag3_smi))[0]
    
    mol = AllChem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles('[4*]'),
                                    Chem.MolFromSmiles(frag4_smi))[0]
    
    return Chem.MolToSmiles(mol)
 

def MolGenForTetra(frags_string, cores_ls):
    
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # split frags_string to three frags
    frags = frags_string.split('.')
    frag1_smi = StandardFrag(frags[0], mode='frag')[1:]
    frag2_smi = StandardFrag(frags[1], mode='frag')[1:]
    frag3_smi = StandardFrag(frags[2], mode='frag')[1:]
    frag4_smi = StandardFrag(frags[3], mode='frag')[1:]

    # corelib
    corelib = [StandardFrag(x, mode='core') for x in cores_ls]

    # generate
    gen_smiles = []
    used_cores = []
    
    for core, core_BRICS_dummy in zip(corelib, cores_ls):
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag2_smi, frag3_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag2_smi, frag4_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag3_smi, frag2_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag3_smi, frag4_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag4_smi, frag2_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag1_smi, frag4_smi, frag3_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag1_smi, frag3_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag1_smi, frag4_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag3_smi, frag1_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag3_smi, frag4_smi, frag1_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag4_smi, frag1_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag2_smi, frag4_smi, frag3_smi, frag1_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag1_smi, frag2_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag1_smi, frag4_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag2_smi, frag1_smi, frag4_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag2_smi, frag4_smi, frag1_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag4_smi, frag1_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag3_smi, frag4_smi, frag2_smi, frag1_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag1_smi, frag2_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag1_smi, frag3_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag2_smi, frag1_smi, frag3_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag2_smi, frag3_smi, frag1_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag3_smi, frag1_smi, frag2_smi))
        gen_smiles.append(ReplaceTetra(core, frag4_smi, frag3_smi, frag2_smi, frag1_smi))
        used_cores += [core_BRICS_dummy] * 24
        
    # Check smiles
    success_smiles, success_cores = [], []
    for smi, core in zip(gen_smiles, used_cores):
        if CheckSMI(smi):
            success_smiles.append(smi)
            success_cores.append(core)

    return success_cores, success_smiles

import json

def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))


def MolGenForPipeline(dict_task, dict_task_frags, mode='single'):
    '''
    mode = 'single', 'double', 'triple', 'tetra'
    '''

    dict_result = {
        'TASKID': [],
        'IDNUMBER': [],
        'main_smi': [],
        'frag_before': [],
        'frag_after': [],
        'SMILES': [],
        'InChiKey': [],
    }

    for task in dict_task.keys():

        main_smi, frag_before = dict_task[task][0], dict_task[task][1]
        frags_ls = dict_task_frags[task]
        
        if mode == 'single':
            success_frags, success_smiles = MolGenForSingle(main_smi, frags_ls)
        elif mode == 'double':
            success_frags, success_smiles = MolGenForDouble(main_smi, frags_ls)
        elif mode == 'triple':
            success_frags, success_smiles = MolGenForTriple(main_smi, frags_ls)
        elif mode == 'tetra':
            success_frags, success_smiles = MolGenForTetra(main_smi, frags_ls)
        else:
            return None

        N = len(success_frags)
        dict_result['TASKID'] += [task] * N
        dict_result['IDNUMBER'] += ['{}-{:06d}'.format(task, i) for i in range(1, N+1)]
        dict_result['main_smi'] += [main_smi] * N
        dict_result['frag_before'] += [frag_before] * N
        dict_result['frag_after'] += success_frags
        dict_result['SMILES'] += success_smiles
        dict_result['InChiKey'] += [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in success_smiles]

    df_result = pd.DataFrame(dict_result)
    
    print('{} site tasks: genmols {}'.format(mode, df_result.shape[0]))

    return df_result

def duplicate_for_genmols(df_genmols):

    unique_id, unique_inchikey = [], []
    for i, x in enumerate(df_genmols.loc[:, 'InChiKey'].values):
        if x not in unique_inchikey:
            unique_id.append(i)
            unique_inchikey.append(x)
    
    df_genmols = df_genmols.loc[unique_id].reset_index(drop=True)
    # print('duplication: genmols {}'.format(df_genmols.shape[0]))

    return df_genmols


if __name__ == '__main__':

    pass
    