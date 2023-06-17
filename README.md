# It Looks Fair, but It's Not
Repository of the paper "It Looks Fair, but It’s Not: Multigroup Fairness Assessment in Graph Neural Network-Based Models for User Profiling" by Erasmo Purificato, Ludovico Boratto and Ernesto William De Luca.

## Abstract
User profiling holds considerable value in various applications, such as information retrieval systems and social networks. Evaluating the effectiveness of a user profiling approach involves assessing its ability to classify personal characteristics such as gender, age, or consumption grade. However, the challenge arises when dealing with multiclass attributes, such as non-binary age, as most fairness metrics typically work with binary attributes. To overcome this challenge, researchers often convert the original multiclass attributes into binary form to align with standard fairness metrics definitions, where both the target class and sensitive attribute (e.g. gender or age) are binary. However, this conversion alters the original conditions, and fairness is evaluated on classes that differ from those used in the classification process.
  In this paper, we propose an extension of four existing fairness metrics related to disparate impact and disparate mistreatment, from binary scenarios to multigroup scenarios that consider non-binary sensitive attributes.
  Our objective is to bridge the gap between formal definitions and real-world use cases in bias detection. We conduct experiments using two real-world datasets and employ two state-of-the-art graph neural network-based models for user profiling. The results demonstrate that our generalized fairness metrics provide a more effective and detailed understanding of disadvantaged sensitive groups. Furthermore, in some cases, they enable a better analysis of machine learning models that were initially perceived as fair.

## Requirements
The code has been executed under **Python 3.8.1**, with the dependencies listed below.

### CatGCN
```
metis==0.2a5
networkx==2.6.3
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.1.2
scipy==1.7.3
texttable==1.6.4
torch==1.10.1+cu113
torch_geometric==2.0.3
torch_scatter==2.0.9
tqdm==4.62.3
```

### RHGN
```
dgl==0.9.1
dgl_cu113==0.7.2
hickle==4.0.4
matplotlib==3.5.1
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.1.2
scipy==1.7.3
torch==1.10.1+cu113
```
Notes:
* the file `requirements.txt` installs all dependencies for both models;
* the dependencies including `cu113` are meant to run on **CUDA 11.3** (install the correct package based on your version of CUDA).

## Datasets
The preprocessed files required for running each model are included as a zip file within the related folder.

The raw datasets are available at:
* **Alibaba**: [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
* **JD**: [link](https://github.com/guyulongcs/IJCAI2019_HGAT)

## Multigroup Fairness Metrics
The repository implements the generalised **Multigroup Fairness Metrics** presented in the paper.

Let:
* $N$ be the number of *demographic groups*;
* $y \in \lbrace 0, 1 \rbrace$ be the *target class*;
* $\hat{y} \in \lbrace 0, 1 \rbrace$ be the *predicted class*;
* $s \in \lbrace 0, ..., N-1 \rbrace$ be the *sensitive attribute*.

The score of each of the metrics displayed below should be equal across every class and group:

### **Multigroup statistical parity**
$$
P(\hat{y} = 1 | s = n), \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multigroup equal opportunity**
$$
P(\hat{y} = 1 | y = 1, s = n), \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multigroup overall accuracy equality**
$$
P(\hat{y} = 0 | y = 0, s = n) + P(\hat{y} = 1 | y = 1, s = n), \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multigroup treatment equality**
$$
\frac{P(\hat{y} = 1 | y = 0, s = n)}{P(\hat{y} = 0 | y = 1, s = n)}, \forall n \in \lbrace 0,...,N-1 \rbrace
$$

## Run the code
Example test runs for each combination of model-dataset.

### CatGCN - Alibaba dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 0.1 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input/ali_data/user_edge.csv \
--field-path ./input_ali_data/user_field.npy --target-path ./input_ali_data/user_buy.csv \
--labels-path ./input_ali_data/user_labels.csv --sens-attr age --label buy
```

### CatGCN - JD dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 1e-2 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 39 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input_jd_data/user_edge.csv \
--field-path ./input_jd_data/user_field.npy --target-path ./input_jd_data/user_expense.csv \
--labels-path ./input_jd_data/user_labels.csv --sens-attr age --label expense
```

### RHGN - Alibaba dataset
```
$ cd RHGN
$ python3 ali_main.py --seed 42 --gpu 0 --model RHGN --data_dir ./input_ali_data/ \
--graph G --max_lr 0.1 --n_hid 32 --clip 2 --n_epoch 100 \
--label bin_buy --sens_attr age
```

### RHGN - JD dataset
```
$ cd RHGN
$ python3 jd_main.py --seed 3 --gpu 0 --model RHGN --data_dir ./input_jd_data/ \
--graph G --max_lr 1e-3 --n_hid 64 --clip 1 --n_epoch 100 \
--label bin_exp --sens_attr age
```

## Contact
<!-- Erasmo Purificato (erasmo.purificato@ovgu.de) -->
erasmo.purificato@ovgu.de