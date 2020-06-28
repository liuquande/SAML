# SAML & Multi-site Data for Prostate MRI Segmentation
by [Quande Liu](https://github.com/liuquande), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

* The Tensorflow implementation for our MICCAI 2020 paper '[Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains](https://github.com/liuquande/SAML)'. 
![](figures/overview.png)

* A well-organized multi-site dataset (from six data sources) for prostate MRI segmentation, that can support research in various problem settings with need of multi-site data, such as [Domain Generalization](https://github.com/amber0309/Domain-generalization), [Multi-site Learning](https://arxiv.org/abs/2002.03366) and [Life-long Learning](https://arxiv.org/abs/1805.10170), etc. The details of sample number and imaging protocols in each site are summarized in the tabel below:

<p align="center">
  <img src="protocol.png"  width="700"/>
</p>

  
For **more details and downloading link** of the orgarized dataset, please [find here](https://liuquande.github.io/SAML/).

#### Setup & Usage for the Code

1. Check dependencies in requirements.txt, and necessarily run:
   ```shell
   pip install -r requirements.txt
   ```
2. To train the model, you need to specify the training configurations (can simply use the default setting) in main.py, then run:
   ```shell
   python main.py --phase=train
   ```

2. To evaluate the model, run:
   ```shell
   python main.py --phase=test --restore_model='/path/to/test_model.cpkt'
   ```
   You will see the output results in the folder `./output/`.

#### The Multi-site Dataset
* For more details and the downloading link of the orgarized dataset, please [find here](https://liuquande.github.io/SAML/).

### Questions

For further question about the code or dataset, please contact 'qdliu@cse.cuhk.edu.hk'
