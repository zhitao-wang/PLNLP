# Pairwise Learning for Neural Link Prediction for OGB (PLNLP-OGB)
This repository provides evaluation codes of PLNLP for OGB link property prediction task. The idea of PLNLP is described in the following article:
>**Pairwise Learning for Neural Link Prediction (https://arxiv.org/pdf/2010.16103.pdf)**

## Environment
The code is implemented with PyTorch and PyTorch Geometric. Requirments:  
&emsp;1. python=3.6  
&emsp;2. pytorch=1.7.1  
&emsp;3. ogb=1.3.2  
&emsp;4. pyg=2.0.1

## Reproduction of performance on OGBL
ogbl-ddi:  

    python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --mlp_num_layers=2 --gnn_num_layers=2 eval_steps=5 --num_neg=3 --epochs=500 --encoder=GraphSage --lr=0.001 --batch_size=65536 --neg_sampler=global --dropout=0.3 
    

## Reference
This work is based on our previous work to some extent. The related works are listed as follow:

