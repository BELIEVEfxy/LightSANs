# LightSANs
This is our Pytorch implementation for our SIGIR 2021 short paper:
> Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, and Ji-Rong Wen (2021). "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." In SIGIR 2021.

# Overview
we propose the low-rank decomposed self-attention networks **LightSANs** to improve the effectiveness and efficiency of SANs-based recommenders. Particularly, it projects user's historical items into a small constant number of latent interests, and leverages item-to-interest interaction to generate the user history representation. Besides, the decoupled position encoding is introduced, which expresses the items’ sequential relationships much more precisely. The overall framework of LightSANs is depicted bellow.

![The framework of LightSANs](https://github.com/BELIEVEfxy/LightSANs/blob/main/model.png)

# Requirements
- Python 3.6
- Pytorch >= 1.3

Notice: For all sequencial recommendation models, we use the first version of RecBole v0.1.1 to do our experiments. The more details are on [RecBole](https://github.com/RUCAIBox/RecBole). For efficient Transformers([Synthesizer](https://github.com/leaderj1001/Synthesizer-Rethinking-Self-Attention-Transformer-Models), [LinTrans](https://linear-transformers.com), [Linformer](https://github.com/tatp22/linformer-pytorch), [Performer](https://github.com/lucidrains/performer-pytorch)), we implement them under RecBole Framework based on the source code, in order to ensure fair comparation. 

# Datasets
We use three real-world benchmark datasets, including Yelp, Amazon Books and ML-1M. The details about full version of these datasets are on [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). For all datasets, we group the interaction records by users and sort them by the interaction timestamps ascendingly. 

# Parameter Settings
We apply the leave-one-out strategy for evaluation, and employ HIT@k and NDCG@k to evaluate the performance. For fair evaluation, we pair each ground truth item in the test set with all items of dataset.

For all SANs-based models, 2 layers of self-attention are deployed, both of which have 2 attention heads. The hidden-dimension of embeddings are set to 64 uniformly. The maximum sequence length is 100, 150 and 200 and the parameter _k_ of LightSANs is 10, 15 and 20 on Yelp, Books and ML-1M datasets, respectively. The dropout rate of turning off neurons is 0.2 for ML-1M and 0.5 for the other four datasets due to their sparsity. The low-rank projected dimension in Synthesizer, Linformer and Performer are set as the same as _k_. We use the Adam optimizer with a learning rate of 0.003 on GPU (TITAN Xp), where the batch size is set as 1024 and 2048 in the training and the evaluation stage, respectively. 

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
If you have any questions for our paper or codes, please send an email to xinyan.fan@ruc.edu.cn.

