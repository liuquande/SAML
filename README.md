# SAML & Multi-site Data for Prostate MRI Segmentation
by [Quande Liu](https://github.com/liuquande), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

* The Tensorflow implementation for our MICCAI 2020 paper '[Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains](https://github.com/liuquande/SAML)'. 
![](figures/overview.png)

* A well-organized multi-site dataset (from six data sources) for prostate MRI segmentation, that can support research in various problem settings with need of multi-site data, such as [Domain Generalization](https://github.com/amber0309/Domain-generalization), [Multi-site Learning](https://arxiv.org/abs/2002.03366) and [Life-long Learning](https://arxiv.org/abs/1805.10170), etc. 

<center><img src="protocol.png" class="centerImage" width="700"/></center>

Among these data:
* Samples of Site A,B are from [NCI-ISBI 2013](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) dataset [1].
* Samples of Site C are from [Initiative  for  Collaborative  Computer  Vision  Benchmarking](https://i2cvb.github.io/) (I2CVB) dataset [2].
* Samples of Site D,E,F are from [Prostate MR Image Segmentation 2012](https://promise12.grand-challenge.org/) (PROMISE12) dataset [3].


### Setup & Usage

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

#### (1) Details of data and imaging protocols of the six different sites included in the organized dataset.

<center><img src="protocol.png" class="centerImage" width="700"/></center>

Among these data:
* Samples of Site A,B are from [NCI-ISBI 2013](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) dataset [1].
* Samples of Site C are from [Initiative  for  Collaborative  Computer  Vision  Benchmarking](https://i2cvb.github.io/) (I2CVB) dataset [2].
* Samples of Site D,E,F are from [Prostate MR Image Segmentation 2012](https://promise12.grand-challenge.org/) (PROMISE12) dataset [3].

#### (2) Preprocessing steps

<br> We frist convert data from all six sites uniformly to '.nii' format. For preprocessing, we first center-cropped the images from Site C with roughly same view  in axial plane as images from other sites, and then clip them in z space to preserve slices with prostate region (since the raw images of Site C are scanned from whole body rather than prostate surrounding area). After that, we resized all samples in the six sites to size of 384x384 in axial plane.

### Acknowledgements

### Reference
\[1\] Bloch, N., Madabhushi, A., Huisman, H., Freymann, J., et al.: NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures. (2015)
<br> \[2\] Lemaitre, G., Marti, R., Freixenet, J., Vilanova. J. C., et al.: Computer-Aided Detection and diagnosis for prostate cancer based on mono and multi-parametric MRI: A review. In: Computers in Biology and Medicine, vol. 60, pp. 8-31 (2015)
<br> \[3\] Litjens, G., Toth, R., Ven, W., Hoeks, C., et al.: Evaluation of prostate segmentation algorithms for mri: The promise12 challenge. In: Medical Image Analysis.
, vol. 18, pp. 359-373 (2014)
