# Direction-based-BiTSE
This repository provides the implementation for the paper "Leveraging Boolean Directivity Embedding for Binaural Target Speaker Extraction" by Yichi Wang, Jie Zhang, Chengqian Jiang, Weitai Zhang, Zhongyi Ye, Lirong Dai.

# Dataset
We utilized the dataset generation method described in https://github.com/huangzj421/BinauralWSJ0Mix.

# Model
For the backbone network, we used the NBC2 small available at https://github.com/audio-westlakeu/nbss.

# Spatiotemporal features
The spatiotemporal features are extracted in spatial_fea.py

#  Paradigm of the proposed BiTSE
![Alt text](images/framework.png)

# Results 
![Alt text](images/table1.png)
![Alt text](images/table2.png)
