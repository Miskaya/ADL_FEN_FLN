# ADL_FEN_FLN
SCAI Semester Project - Robust Classification of the Activities of Daily Living using Unobtrusive Sensors and Transfer Learning
==========

Overview
==========
The goal is to effectively classify human activities. The study introduces a novel pipeline capable of
transforming any kinematic signal associated with human motion into a robust feature representation. We outperformed numerous state-of-the-art models across 11 public Human Activity Recognition (HAR) datasets. Through comprehensive comparisons with baseline models on multiple datasets, we established the superiority of our proposed pipeline. Notably, our investigation extends beyond existing benchmarks by evaluating our pipeline on sensei-v2 dataset. This dataset not only represents a distinct population but also incorporates diverse kinematic modalities. The results underscore the pipeline’s remarkable generalization capabilities in feature representation of human movement.

Run Program
========

### Initialize the Working Environment
To be able to run this program, the user must first set up a `conda` environment. 
The user must have [Anaconda](https://www.anaconda.com/) installed on their device. 
To create a new environment, run the following command on a terminal: 
```
conda env create --name bp_estimation
```

Activate the created environment by running the following command: 

```
conda activate ADL_FEN_FLN
```

The user must install the correct packages to run the Jupyter notebooks with the command: 
```
conda install <PACKAGE>
```

The user can also directly create an environment by running the command:
```
conda env create -f requirements.yml

```

The user will have to start a Jupyter notebook server to be able to run the `.ipynb` files. 
This is done either by openning the application of choice (the application used for coding was [Visual Studio Code](https://code.visualstudio.com)) or the Jupyter Notebook browser of Anaconda with: 

```
jupyter notebook
```
### Data Preprocessing
The 'Data_Generated.ipynb' does the data preprocessing for all 11 public HAR datasets. Each signal will be one column and the last two columns are subjects and labels. All the datasets will be resampled to 20Hz with standardization. Each dataset will be saved to a pickle file.

### Best Hyper-parameters
The 'Dataset_params.py' shows all the hyper-parameters for 11 public datasets. It is from hyper-parameter tuning and be chosen with best classification results for each dataset.

### Model
The 'Model.py' is the Feature Extraction Network (FEN) and Feature Learning Network (FLN) model.

### Training Process
The 'Train.py' is the training process for all 11 public datasets.

### Transfer learning
The 'Transfer_learning.py' uses transfer learning on the sensei-v2 dataset with Best saved FEN weights to get classification results.

