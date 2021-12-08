# Pairwise Learning for Neural Link Prediction (PLNLP)
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

    python train.py -d data/meme -l 0.001 -x 32 -e 64 -b 32 -t 1 
    

## Reference
This work is based on our previous work to some extent. The related works are listed as follow:

