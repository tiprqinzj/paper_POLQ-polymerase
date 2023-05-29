# Data Availability Statement

This repo is the **Data Availability Statement** part of the manuscript entitled **"Ligand- and Structure-Based Machine Learning Models for Predicting Bioactivities of DNA Polymerase Theta Inhibitors"**. As the submitted or revised status of the manuscript, the repo files may change  from time to time.

## Mainly required packages version

 - RDKit: 2020.09.1.0
 - NumPy: 1.21.5
 - Scikit-Learn: 0.24.2
 - Maestro: release 2020-3

## Instructions of repo files

In folder **data**, there are eight files in it.

 - **7zx1_A_glide-grid.zip**: The receptor grid file obtained from Maestro Glide tool;
 - **File2_dataset123.csv**: The 298 inhibitors of the datasets 1, 2, and 3. Same to the Supporting Information of the manuscript;
 - **File3_dataset4.csv**: The 325 inhibitors of the dataset4. Same to the Supporting Information of the manuscript;
 - **File4_PaDEL2D_descriptors_list.txt**: The calculated 784 PaDEL 2D molecular descriptors. Same to the Supporting Information of the manuscript;
 - **File5_RDKit2D_descriptors_list.txt**: The calculated 208 RDKit 2D descriptors. Same to the Supporting Information of the manuscript;
 - **File6_pretreated_7zx1.pdb**: The pretreated POLQ protein and the ligands interacting with it (ART558, DNA, GTP, ART558, and Mg2+). Same to the Supporting Information of the manuscript;
 - **File7_BRICS_substructure_library.csv**: The BRICS substructure library file, which has 45619 substructures. Same to the Supporting Information of the manuscript;
 - **File8_generatedmols_predscores.csv**: The predicted scores of 24260 generated molecules. Same to the Supporting Information of the manuscript.

In folder **models**, there are two folders in it.

 - **Model2**: containing 30 sub-models, which build consensus Model 2;
 - **Model2**: containing 15 sub-models, which build consensus Model 4.

In folder **scripts**, there are ten coding script in it.

 - **S01_preprocessing.py**: check SMILES, duplicate, flatten (details in *Data Collection and Curation* of the manuscript);
 - **S02_trtesplit.py**: stratified split to training and external test sets (details in *Data Collection and Curation* of the manuscript);
 - **S03_validation.py**: stratified split the training set to equal parts (details in *Data Collection and Curation* of the manuscript);
 - **S04_modeling.py**: random forest workflows (details in *Ligand-based Machine Learning Models* of the manuscript);
 - **S05_consensus.py**: build consensus models from several sub-models (details in *Ligand-based Machine Learning Models* of the manuscript);
 - **S06_consensus.py**: BRICS molecular generation workflows (details in *Molecular Generation* of the manuscript);
 - **cal_fingerprint.py**: imported funtions for *S04_modeling.py*;
 - **cal_descriptor.py**: imported funtions for *S04_modeling.py*;
 - **modeling_rfc_fp.py**: imported funtions for *S04_modeling.py*;
 - **modeling_rfc_num.py**: imported funtions for *S04_modeling.py*.

