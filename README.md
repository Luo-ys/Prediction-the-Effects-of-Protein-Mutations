
## A Fusion model for Prediction the Effects of Protein Mutations 

This project investigated the impact of protein mutations through a deep learning-based method that utilizes multi-scale structural properties of proteins. The synergistic integration of protein co-evolution, sequence semantics, and geometric features were used for improved protein mutation effect prediction.

<img src="./fig/fig.jpg">

## Install

Install this package
```bash
pip install fair-esm torch torchdrug scikit-learn tensorflow-gpu numpy2tfrecord
```
## Export protein features

Running this code (**precess-seq-data-tf.ipynb**) gets the sequence features.
The .braw format files can be accessed by HHblits and CCMPred (https://github.com/luoyunan/ECNet).

Then, running this code (**process-structure-data.ipynb**) gets the structure features.
The protein structure informations come from the Gearnet (https://github.com/DeepGraphLearning/GearNet).

## Training

```bash
run train-tf-fina.ipynb
```


## Reference
```bibtex
@article{luo2021ecnet,
  doi = {10.1038/s41467-021-25976-8},
  url = {https://doi.org/10.1038/s41467-021-25976-8},
  year = {2021},
  month = sep,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {12},
  number = {1},
  author = {Yunan Luo and Guangde Jiang and Tianhao Yu and Yang Liu and Lam Vo and Hantian Ding and Yufeng Su and Wesley Wei Qian and Huimin Zhao and Jian Peng},
  title = {{ECNet} is an evolutionary context-integrated deep learning framework for protein engineering},
  journal = {Nature Communications}
}
```

```bibtex
@article{zhang2023enhancing,
  title={A Systematic Study of Joint Representation Learning on Protein Sequences and Structures},
  author={Zhang, Zuobai and Wang, Chuanrui and Xu, Minghao and Chenthamarakshan, Vijil and Lozano, Aurelie and Das, Payel and Tang, Jian},
  journal={arXiv preprint arXiv:2303.06275},
  year={2023}
}
```
