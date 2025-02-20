# GNN4siRNA

## Installation and requirements
### requirements
- ViennaRNA v2.7.0
- python 3.6
  - anaconda or pyenv installed

### Installation
```
conda create --name rnai python=3.6
conda activate rnai
```
or  
```
pyenv install 3.6.13
pyenv virtualenv 3.6.13 rnai
pyenv activate rnai
```
After within the new virtual envirnment:  
`pip install -r requirements.txt`

## Model Creation & Prediction

The original code lacked functionality for creating the model and making predictions on new data. Due to compatibility challenges with StellarGraph, saving and reloading the model for later predictions caused multiple issues.

To address this, the provided script creates the model and performs predictions within the same execution. Since model creation is not computationally intensive, this approach ensures smooth operation. Additionally, model optimization has been incorporated into the original model generation. The most time-consuming step will be running ViennaRNA.

## Usage

Run the following command within the previously created environment:  
  
`python predict.py --sirna_fasta <siRNA_fasta_file> --mrna_fasta <mRNA_fasta_file> --sirna_mrna_csv <csv_file>`  
  
Where `<csv_file>` is a CSV file with two columns:

- siRNA: The siRNA name as used in the FASTA file.
- mRNA: The corresponding mRNA target name from the mRNA FASTA file.

This ensures correct pairing between siRNA sequences and their target mRNAs for prediction.  

## Original text

GNN approach to face with the problem of siRNA-mRNA efficacy prediction.

This repository provides the source code for "A graph neural network approach for the analysis of siRNA-target biological networks".

- The "**data**" Folder contains both "raw" and "processed" data. We reported three siRNA-mRNA interaction network datasets from 702 to 3518 interactions.

- The "**preprocessing**" folder contains five scripts we released for transforming raw data into the input of our model. It also includes a "params.py" file with paths of input files and other parameters.

- The "**model**" folder contains the main script "GNN4siRNA.py" for predicting siRNA-mRNA interactions. Once again, the "params.py" file contains all the parameters reported in the paper. Results of this script are given in terms of *Pearson correlation coefficient* and *mean squared error*.
