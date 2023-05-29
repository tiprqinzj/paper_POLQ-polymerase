import pandas as pd
from rdkit import RDLogger, Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Descriptors import ExactMolWt


def check_mol(smi):
    
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol
        else:
            return False
    except:
        return False

def remove_frag(mol):

    remover = rdMolStandardize.FragmentRemover()

    mol = remover.remove(mol)
    mol = rdMolStandardize.FragmentParent(mol)

    return mol


def check_heavy_atoms(mol, MIN=10, MAX=60):

    HA = mol.GetNumHeavyAtoms()

    if HA < MIN or HA > MAX:
        return (HA, False)
    else:
        return (HA, True)

# def check_MolWt(mol, MIN=295.7, MAX=648.5):

#     MW = round(ExactMolWt(mol), 2)

#     if MW < MIN or MW > MAX:
#         return (MW, False)
#     else:
#         return (MW, True)

def check_MolWt(mol, MEAN=472.1, STD=88.2):
    '''
    modify in 2022.12.01: from MAX and MIN to MEAN and STD
    '''

    MW = round(ExactMolWt(mol), 2)

    if MW < (MEAN - 3 * STD) or MW > (MEAN + 3 * STD):
        return (MW, 'Out 3STD')
    elif MW < (MEAN - 2 * STD) or MW > (MEAN + 2 * STD):
        return (MW, 'Out 2STD')
    else:
        return (MW, 'Pass')


def check_symbol(mol, common_symbols=['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si']):

    all_symbols = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in all_symbols:
            all_symbols.append(atom.GetSymbol())
    
    if 'C' not in all_symbols:
        return (False, 'Delete because no Carbon in mol')
    
    others = []
    for s in all_symbols:
        if s not in common_symbols:
            others.append(s)
    
    if len(others) > 0:
        return False
    else:
        return True


def pipeline(in_smiles_txt, out_results_txt):
    
    print('Start calculation:')

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # result_dict
    d_out = {
        'check_tag': [],
        'check_smiles': [],
        'check_symbol': [],
        'HA': [],
        'HA_tag': [],
        'MW': [],
        'MW_tag': [],
        'inchikey': [],
        'inchikey14': [],
        'flatten_smiles': [],
    }


    # load data
    with open(in_smiles_txt) as f:
        smiles_in = [line.strip() for line in f]

    # check smi
    success_count, Symbol_count, HA_count = 0, 0, 0

    for smi in smiles_in:

        mol = check_mol(smi)

        if mol == False:
            d_out['check_tag'].append('Fail')
            d_out['check_smiles'].append('')
            d_out['check_symbol'].append('')
            d_out['HA'].append('')
            d_out['HA_tag'].append('')
            d_out['MW'].append('')
            d_out['MW_tag'].append('')
            d_out['inchikey'].append('')
            d_out['inchikey14'].append('')
            d_out['flatten_smiles'].append('')
            continue

        try:
            mol = remove_frag(mol)
        except:
            d_out['check_tag'].append('Fail')
            d_out['check_smiles'].append('')
            d_out['check_symbol'].append('')
            d_out['HA'].append('')
            d_out['HA_tag'].append('')
            d_out['MW'].append('')
            d_out['MW_tag'].append('')
            d_out['inchikey'].append('')
            d_out['inchikey14'].append('')
            d_out['flatten_smiles'].append('')
            continue

        # check_tag & check_smiles
        smi_out = Chem.MolToSmiles(mol)
        d_out['check_tag'].append('Pass')
        d_out['check_smiles'].append(smi_out)
        success_count += 1


        # check symbol
        if check_symbol(mol):
            d_out['check_symbol'].append('Pass')
            Symbol_count += 1
        else:
            d_out['check_symbol'].append('Fail')


        # check Heavy Atoms and MolWt
        HA, HA_tag = check_heavy_atoms(mol)
        if HA_tag:
            d_out['HA'].append(HA)
            d_out['HA_tag'].append('Pass')
            HA_count += 1
        else:
            d_out['HA'].append(HA)
            d_out['HA_tag'].append('Fail')
        
        MW, MW_tag = check_MolWt(mol)
        d_out['MW'].append(MW)
        d_out['MW_tag'].append(MW_tag)
        
        # InchiKey
        inchikey = Chem.MolToInchiKey(mol)
        d_out['inchikey'].append(inchikey)
        d_out['inchikey14'].append(inchikey[:14])


        # Flatten
        smi_flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        d_out['flatten_smiles'].append(smi_flat)

    print('Input: {}'.format(len(smiles_in)))
    print('Success: {}'.format(success_count))
    print('Passed Symbol, Heavy Atoms: {}, {}'.format(Symbol_count, HA_count))
        

    # save
    df_out = pd.DataFrame(d_out)
    df_out.columns = ['checked', 'Checked SMILES', 'symbol', 'HA', 'HA_tag', 'MolWt', 'MolWt_tag', 'InChiKey', 'InChiKey14', 'Flatten SMILES']
    df_out.to_csv(out_results_txt, index=None, sep='\t')
    print('Done.')


if __name__ == '__main__':

    pipeline(
        in_smiles_txt   = '/home/cadd/Project/TY8103_PKMYT1/20230508_pred_caco2/S1_prepare.smi',
        out_results_txt = '/home/cadd/Project/TY8103_PKMYT1/20230508_pred_caco2/S2_prepare_out.txt',
    )
