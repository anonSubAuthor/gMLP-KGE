# gMLP-KGE: A Simple but Efficient MLPs With Gating Architecture for Link Prediction

This is the code of paper 
**gMLP-KGE: A Simple but Efficient MLPs With
Gating Architecture for Link Prediction**. 

## Requirements
- python 3.8
- torch 1.9
- dgl 0.8

## Available datasets are:
    
    FB15k-237
    WN18RR
    FB15K
    WN18
    DB100K
    YAGO3-10
    Kinship
    


## Reproduce the Results
To run a model execute the following command :
- FB15k-237

```python run.py --data FB15k-237 --g_drop 0.3 --s_drop 0.0 --input_drop 0.3 --feature_drop 0.3 --hidden_drop 0.3 --init_dim 200 --gcn_dim 200 --embed_dim 200 --k_h 400 --t 0.007```
- WN18RR

```python run.py --data wn18rr --g_drop 0.3 --s_drop 0.0  --input_drop 0.3 --feature_drop 0.1 --hidden_drop 0.4  --init_dim 200 --gcn_dim 200 --embed_dim 200 --k_h 400 --t 0.007```

- FB15K
    
```python run.py --data FB15k --g_drop 0.0 --s_drop 0.0 --input_drop 0.2 --feature_drop 0.2 --hidden_drop 0.2 --init_dim 200 --gcn_dim 200 --embed_dim 200 --k_h 400 --t 0.001```
- WN18
 
```python run.py --data wn18 --g_drop 0.0 --s_drop 0.0 --input_drop 0.2 --feature_drop 0.3 --hidden_drop 0.3 --init_dim 200 --gcn_dim 200 --embed_dim 200 --k_h 400 --t 0.1```
- DB100K
   
```python run.py --data DB100K --g_drop 0.3 --s_drop 0.0 --input_drop 0.3 --feature_drop 0.3 --hidden_drop 0.3 --init_dim 300 --gcn_dim 300 --embed_dim 300 --k_h 600 --t 0.03```
- YAGO3-10
    
```python run.py --data yago --g_drop 0.0 --s_drop 0.0 --input_drop 0.2 --feature_drop 0.3 --hidden_drop 0.3 --init_dim 300 --gcn_dim 300 --embed_dim 300 --k_h 600 --t 0.0005```

- Kinship

```python run.py --data kinship --g_drop 0.6 --s_drop 0.0 --input_drop 0.2 --feature_drop 0.3 --hidden_drop 0.3 --init_dim 200 --gcn_dim 200 --embed_dim 200 --k_h 400 --t 1.0```




## Acknowledgement
We refer to the code of [LTE](https://github.com/MIRALab-USTC/GCN4KGC) and [DGL](https://github.com/dmlc/dgl). Thanks for their contributions.
