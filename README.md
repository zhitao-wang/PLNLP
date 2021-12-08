# Pairwise Learning for Neural Link Prediction (PLNLP)
This repository provides evaluation codes of PLNLP for OGB link property prediction task. The idea of PLNLP is described in the following article:
>**Pairwise Learning for Neural Link Prediction (https://arxiv.org/pdf/2010.16103.pdf)**

## Environment
The code is implemented with PyTorch and PyTorch Geometric. Requirments:  
&emsp;1. Python 2.7  
&emsp;2. Numpy  
&emsp;3. Tensorflow  
&emsp;4. tqdm (for training process display)   

## Run
Defalut:  

    python train.py  
    
Or run with optional arguments:  

    python train.py -d data/meme -l 0.001 -x 32 -e 64 -b 32 -t 1
Check the arguments as:  

    python train.py -h
    -l, --lr (learning rate)  
    -x, --xdim (embedding dimension)  
    -e, --hdim (hidden dimension)  
    -d, --data (data path)  
    -g, --gpu (gpu id)  
    -b, --bs (batch size)  
    -t, --tu (time unit)  

You can also change values of training parameters in "Class Config()" in "train.py"

## Data Format
### Cascades:
>u1_id,u2_id,...,ul_id:u1_time,u2_time,...,ul_time   
>Example: 334,478,398,222:75.117015,77.968750,78.757250,80.020426

## Citing
    @inproceedings{ijcai2019-531,
      title     = {Hierarchical Diffusion Attention Network},
      author    = {Wang, Zhitao and Li, Wenjie},
      booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
                   Artificial Intelligence, {IJCAI-19}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},             
      pages     = {3828--3834},
      year      = {2019},
      month     = {7},
      doi       = {10.24963/ijcai.2019/531},
      url       = {https://doi.org/10.24963/ijcai.2019/531},
    }
