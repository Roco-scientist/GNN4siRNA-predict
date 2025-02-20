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

## Running
The original code had no way of creating the model and predicting on new data.  Since stellargraph is used, creating and saving the model, then predicting later was causing too many issues.
Therefor with the following script the model is created, then used to predict within the same run.  Model creation is not too computationally intensive.  Additional model optimization was added to the original model generation.
ViennaRNA will take the most time.  To run while in the environment created above:
`python predict.py --sirna_fasta <siRNA_fasta_file> --mrna_fasta <mRNA_fasta_file> --sirna_mrna_csv <csv_file>`  
Where `<csv_file>` is a file with two columns `siRNA,mRNA` where the siRNA name used within the fasta file is in the first column and the corresponding mRNA target name from the mRNA fasta file is in the second column.  
## Original text

GNN approach to face with the problem of siRNA-mRNA efficacy prediction.

This repository provides the source code for "A graph neural network approach for the analysis of siRNA-target biological networks".

- The "**data**" Folder contains both "raw" and "processed" data. We reported three siRNA-mRNA interaction network datasets from 702 to 3518 interactions.

- The "**preprocessing**" folder contains five scripts we released for transforming raw data into the input of our model. It also includes a "params.py" file with paths of input files and other parameters.

- The "**model**" folder contains the main script "GNN4siRNA.py" for predicting siRNA-mRNA interactions. Once again, the "params.py" file contains all the parameters reported in the paper. Results of this script are given in terms of *Pearson correlation coefficient* and *mean squared error*.
