# Label-Efficient Interactive Time-Series Anomaly Detection

This repository is for demonstration of our work, "Label-Efficient Interactive Time-Series Anomaly Detection".

[Arxiv](https://arxiv.org/abs/2212.14621)

# Setup

## Requirements

```
conda env create -f environment.yml
```

## Dataset

```
unzip yahoo.zip
```

The KPI dataset can be downloaded at [here](https://github.com/NetManAIOps/KPI-Anomaly-Detection)

## Proprocess

You can download the preprocessed Yahoo dataset at [here](https://drive.google.com/file/d/1llcD6ofi0Gp5ufwwkK2JXsScEZ4RAAhP/view?usp=sharing).

Also you can DIY with your own data:

**1. Prepare the dataset**

```
python data/gen_dataset.py --dataset yahoo
```

**2. Run unsupervised methods**

```
python uad.py --dataset yahoo
```

**3. Train Ts2Vec model and get the representation**

```
python run_ts2vec.py --dataset yahoo --train
```

**4. Generate features**

```
python preprocess_features.py --dataset yahoo
```

Train and test set are split by defining ```data/{dataset}_train.txt``` and ```data/{dataset}_test.txt```.

# BibTeX

```
@article{guo2022label,
  title={Label-Efficient Interactive Time-Series Anomaly Detection},
  author={Guo, Hong and Wang, Yujing and Zhang, Jieyu and Lin, Zhengjie and Tong, Yunhai and Yang, Lei and Xiong, Luoxing and Huang, Congrui},
  journal={arXiv preprint arXiv:2212.14621},
  year={2022}
}
```


