# Pairwise Learning for Neural Link Prediction for OGB (PLNLP-OGB)
This repository provides evaluation codes of PLNLP for OGB link property prediction task. The idea of PLNLP is described in the following article:
>**Pairwise Learning for Neural Link Prediction (https://arxiv.org/pdf/2112.02936.pdf)**

The performance of PLNLP on OGB link prediction tasks is listed as the following tables:
||   ogbl-ddi (Hits@20)   |   ogbl-collab (Hits@50)   |  ogbl-citation2 (MRR)  |
|  ----  |  ----  | ----  | ----  |
|  Validation | 82.42 ± 2.53  | 100.00 ± 0.00 | 84.90 ± 0.31 |
|  Test | 90.88 ± 3.13  | 68.72 ± 0.52 | 84.92 ± 0.29 |

Only with basic graph neural layers (GraphSAGE or GCN), PLNLP achieves **Top-1** performance on ogbl-ddi, and **Top-2** on both ogbl-collab and ogbl-citation2 in current OGB Link Property Prediction Leader Board until **Dec 11, 2021** (https://ogb.stanford.edu/docs/leader_linkprop/), which demonstrates the effectiveness of the proposed framework. We beielve that the performance will be further improved with link prediction specific neural architecure, such as proposed ones in our previous work [2][3]. We leave this part in the future work.

## Environment
The code is implemented with PyTorch and PyTorch Geometric. Requirments:  
&emsp;1. python=3.6  
&emsp;2. pytorch=1.7.1  
&emsp;3. ogb=1.3.2  
&emsp;4. pyg=2.0.1

## Reproduction of performance on OGBL
ogbl-ddi:  

    python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --epochs=500 --neg_sampler=global --dropout=0.3 

ogbl-collab: 

Validation set is allowed to be used for training in this dataset. Meanwhile, following the trick of HOP-REC, we only use training edges after year 2010 with validation edges, and train the model on this subgraph. 

    python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --num_neg=1 --epochs=800 --eval_last_best=True --neg_sampler=global --dropout=0.3

ogbl-citation2:  

    python main.py --data_name=ogbl-citation2 --use_node_feat=True --encoder=GCN --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --dropout=0 

## Reference
This work is based on our previous work as listed below:

[1] Zhitao Wang, Chengyao Chen, Wenjie Li. "Predictive Network Representation Learning for Link Prediction" (SIGIR'17) [[Paper](https://zhitao-wang.github.io/paper/pnrl.pdf)]

[2] Zhitao Wang, Yu Lei and Wenjie Li. "Neighborhood Interaction Attention Network for Link Prediction" (CIKM'19) [[Paper](https://dl.acm.org/doi/10.1145/3357384.3358093)]

[3] Zhitao Wang, Yu Lei and Wenjie Li. "Neighborhood Attention Networks with Adversarial Learning for Link Prediction " (TNNLS) [[Paper](https://ieeexplore.ieee.org/document/9174790)]


