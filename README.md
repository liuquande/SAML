# SAML & Multi-site Data for Prostate MRI Segmentation
by [Quande Liu](https://github.com/liuquande), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

* The Tensorflow implementation for our MICCAI 2020 paper '[Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains](https://github.com/liuquande/SAML)'. 
![](figures/overview.png)

* A well-organized multi-site dataset (from six data sources) for prostate MRI segmentation, that can support research in various problem settings with need of multi-site data, such as [Domain Generalization](https://github.com/amber0309/Domain-generalization), [Multi-site Learning](https://arxiv.org/abs/2002.03366) and [Life-long Learning](https://arxiv.org/abs/1805.10170), etc. The details of sample number and imaging protocols in each site are summarized in the tabel below:

<p align="center">
  <img src="protocol.png"  width="700"/>
</p>

### Setup & Usage for the code

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

### Questions

For further question about the code or dataset, please contact 'qdliu@cse.cuhk.edu.hk'
### A Multi-site Dataset for Prostate MRI Segmentation
The well-organized multi-site data for public use, contrusted from three public datasets of prostate MRI segmentation. The potential research field can cover [Domain Generalization](https://github.com/liuquande/SAML), [Multi-site Learning](https://arxiv.org/abs/2002.03366) and [Life-long Learning](https://arxiv.org/abs/1805.10170), etc.

### Reference
\[1\] Bloch, N., Madabhushi, A., Huisman, H., Freymann, J., et al.: NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures. (2015)
<br> \[2\] Lemaitre, G., Marti, R., Freixenet, J., Vilanova. J. C., et al.: Computer-Aided Detection and diagnosis for prostate cancer based on mono and multi-parametric MRI: A review. In: Computers in Biology and Medicine, vol. 60, pp. 8-31 (2015)
<br> \[3\] Litjens, G., Toth, R., Ven, W., Hoeks, C., et al.: Evaluation of prostate segmentation algorithms for mri: The promise12 challenge. In: Medical Image Analysis.
, vol. 18, pp. 359-373 (2014)
