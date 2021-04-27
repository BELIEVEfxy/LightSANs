# LightSANs
This is our Pytorch implementation for our SIGIR 2021 short paper:
> Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, and Ji-Rong Wen (2021). "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." In SIGIR 2021.

# Introduction

# Requirements
- Python 3.6
- Pytorch >= 1.3

# Datasets
We use three real-world benchmark datasets, including Yelp, Amazon Books and ML-1M. The details about full version of these datasets are on [this link](https://github.com/RUCAIBox/RecSysDatasets). For all datasets, we group the interaction records by users and sort them by the interaction timestamps ascendingly. 

# Parameter Settings
We apply the leave-one-out strategy for evaluation, and employ HIT@$k$ and NDCG@$k$ to evaluate the performance. For fair evaluation, we pair each ground truth item in the test set with all items of dataset.

For all SANs-based models, 2 layers of self-attention are deployed, both of which have 2 attention heads. The hidden-dimension of embeddings are set to 64 uniformly. The maximum sequence length is 100, 150 and 200 and the parameter $k$ of LightSANs is 10, 15 and 20 on Yelp, Books and ML-1M datasets, respectively. The low-rank projected dimension in Synthesizer, Linformer and Performer are set as the same as $k$. We use the Adam optimizer with a learning rate of 0.003 on GPU (TITAN Xp), where the batch size is set as 1024 and 2048 in the training and the evaluation stage, respectively. 

More details about the settings are in .yaml files in properties/dataset and properties/model.

# Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
````
@inproceedings{Fan-SIGIR-2021,
    title = "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation",
    author = {Xinyan Fan and
              Zheng Liu and
              Jianxun Lian and
              Wayne Xin Zhao and
              Xing Xie and 
              Ji{-}Rong Wen},
    booktitle = {{SIGIR}},
    year = {2021},
}
````


