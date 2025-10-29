# 책 부록 소스 프로젝트 입니다

- 직무 교육( OJT, On the job Training )을 위해서 생성.  
- 진행중( WIP, Work on Progress ).  
  + ...


## 책 관련 링크  

<img src="https://image.aladin.co.kr/product/29056/6/cover500/1617296481_2.jpg" alt="" height="256px" align="right">

- [Math and Architectures of Deep Learning [ 원서 ]](https://www.manning.com/books/math-and-architectures-of-deep-learning)  

- [Math and Architectures of Deep Learning [ 외국도서 ]](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=290560627) 

- [Math and Architectures of Deep Learning [ 번역서 없음 ]](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=290560627)  


## 개발 환경 구축

- 시스템 ( Computer System )  

  - AMD Ryzen 9 7900X 12-Core Processor
  - 32G RAM
  - NVIDIA Geforce RTX 3060 12GB
  - SSD 2TB
  - Windows 11 64bit Korean

- 파이썬 ( Python 3.12 )  

  - [Python Download](https://www.python.org/downloads/)  
    - [v3.12.0 for Windows](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe)  
    - [v3.11.9 for Windows](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)  

- ...  


## 의존 패키지

**다음과 같은 순서로 설치하세요.**
  
```
$ (.venv) pip install ipykernel numpy matplotlib scipy  
$ (.venv) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
$ (.venv) pip install lightning
```

**모듈정보**
- numpy
  - [pypi](https://pypi.org/project/numpy/)  
    ```
    $ (.venv) pip install numpy
    ```
  - Fundamental Package for Array Computing in Python

- matplotlib
  - [pypi](https://pypi.org/project/matplotlib/)  
    ```
    $ (.venv) pip install matplotlib
    ```
  - Python Plotting Package

- scipy  
  - [pypi](https://pypi.org/project/scipy/)  
    ```
    $ (.venv) pip install scipy
    ```
  - Fundamental algorithms for scientific computing in Python

- scikit-learn
  - [pypi](https://pypi.org/project/scikit-learn/)  
    ```
    $ (.venv) pip install matplotlib
    ```
  - A set of python modules for machine learning and data mining

- ipykernel
  - [pypi](https://pypi.org/project/ipykernel/)  
    ```
    $ (.venv) pip install ipykernel
    ```
  - [ipykernel](https://github.com/ipython/ipykernel)  
  - IPython Kernel for Jupyter

- PyTorch
  - [pypi](https://pypi.org/project/torch/)  
    ```
    $ (.venv) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```
  - [PyTorch](https://pytorch.org/)  
  - Tensors and Dynamic neural networks in Python with strong GPU acceleration
  - nvidia-smi v531.15
  - cuda-toolkit v12.8

- PyTorch Lightning
  - [pypi](https://pypi.org/project/lightning/)  
    ```
    $ (.venv) pip install lightning
    ```
  - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)  
  - The deep learning framework to pretrain, finetune and deploy AI models  
  - PyTorch Lightning is just organized PyTorch - Lightning disentangles PyTorch code to decouple the science from the engineering.
  - ...  

- ...
  - [pypi]()  
    ```
    $ (.venv) pip install ...
    ```
  - [...]()
  - ...  


## 기타

- ...  

- ...  



---
---
---




# Math and Architectures of Deep Learning

Python code (in the form of Jupyter ipython notebooks) to support the book "**Math and Architectures of Deep Learning**".

Code contributors: **Ananya Ashok, Sujay Narumanchi, Devashish Shankar, Krishnendu Chaudhury**.

This repository contains the example code - mostly in Numpy and PyTorch - corresponding to
the theoretical topics introduced in the book. The code listings are organized in chapters
that correspond to the main book.

## Installation
1. Clone the repository: `git clone https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython.git`
2. Create virtual environment: `virtualenv venv --python=python3` (you may need to do  `pip install virtualenv` first)
3. Activate virtual environment: `source venv/bin/activate` 
4. Change directory: `cd mathematical-methods-in-deep-learning-ipython`
5. Install dependencies: `pip install -r requirements.txt`
6. Navigate to the python directory: `cd python`
7. Start jupyter: `jupyter notebook`

This will redirect you to a browser window with the ipython notebooks 

Note: Ensure to use Python3 to run the notebooks

## Table of Contents

* Chapter 2:
  * [2.2 Intro to Vectors](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.2-vector-pytorch-intro.ipynb)
  * [2.4 Intro to Matrices, Tensors and Images](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.4-matrix-pytorch-intro.ipynb)
  * [2.7 Basic Vector and Matrix operations](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.7-transpose-dot-matmul.ipynb)
  * [2.12.5 Solving an overdetermined system using pseudo inverse](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.12.5-overdet.ipynb)
  * [2.13 Eigenvalues and Eigenvectors](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.13-eig.ipynb)
  * [2.14 Rotation Matrices](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.14-rotation.ipynb)
  * [2.15 Matrix Diagonalization](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.15-mat-diagonalization.ipynb)
  * [2.16 Spectral Decomposition of a Symmetric Matrix](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.16-spectral-decomp.ipynb)
  * [2.17 Finding the axes of a hyper-ellipse](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.17-hyper-ellipse.ipynb)
  
  
* Chapter 3
  * [3.4 Common code for chapter 3](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4-common.ipynb)
  * [3.4.1 Gradient Descent](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.1-gradients.ipynb)
  * [3.4.2 Non-linear Models](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.2-gradients-nonlinear.ipynb)
  * [3.4.3 A Linear Model for the cat-brain](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.3-gradients-catbrain.ipynb)
  
* Chapter 4
  * [4.3.2 Common code for chapter 4](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-common.ipynb)
  * [4.3.2 PCA on synthetic correlated data](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-pca.ipynb)
  * [4.3.2 PCA on synthetic uncorrelated data](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-pca-uncorrelated.ipynb)
  * [4.3.3 PCA on synthetic correlated non-linear data](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.3-pca-nonlinear.ipynb)
  * [4.4.4 Linear system solving via SVD](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.4.4-svd-linear-system.ipynb)
  * [4.4.5 PCA computation via SVD](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.4.5-svd-pca.ipynb)
  * [4.5.3 LSA on a toy dataset](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.5.3-svd-lsa-toy-dataset.ipynb)
  * [4.5.4 LSA/SVD on a 500 × 3 dataset](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.5.4-svd-lsa.ipynb)

* Chapter 5
  * [5.9.1 Uniform Random Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.1-uniform-random-distribution.ipynb)
  * [5.9.2 Normal Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.2-normal-distribution.ipynb)
  * [5.9.3 Binomial Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.3-binomial-distribution.ipynb)
  * [5.9.4 Multinomial Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.4-multinomial-distribution.ipynb)
  * [5.9.5 Bernoulli Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.5-bernoulli-distribution.ipynb)
  
* Chapter 6
  * [6.2.2 Entropy](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.2.2-entropy-gaussian.ipynb)
  * [6.3.1 Cross Entropy](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.3.1-cross-entropy.ipynb)
  * [6.4.2 KL Divergence](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.4.2-kullback-leibler-divergence.ipynb)
  * [6.7.1 Model Parameter Estimation](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.7.1-model-parameter-estimation.ipynb)
  * [6.8.2 Gaussian Mixture Modelling](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.8.2-gaussian-mixture-models.ipynb)

* Chapter 7
  * [7.2.3 Perceptron](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.2.3-perceptron.ipynb)
  * [7.2.4 Modeling logic gates with perceptrons](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.2.4-modeling-logic-gates-with-perceptrons.ipynb)
  * [7.4.3 Approximating surfaces with perceptrons](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.4.3-approximating-surfaces-with-perceptrons.ipynb)

* Chapter 8
  * [8.4 Forward and Backward pass](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch8/8.4-forward-and-backward-pass.ipynb)
  * [8.5 Training a neural network](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch8/8.5-training-a-neural-network.ipynb)

* Chapter 9
  * [9.1 Loss Functions](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.1-loss-functions.ipynb)
  * [9.2.3 Stochastic Gradient Descent](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.2.3-stochastic-gradient-descent.ipynb)
  * [9.2.4 Momentum](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.2.4-momentum.ipynb)

* Chapter 10
  * [10.3.1 2D Convolution for image smoothing](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.3.1-2dconv-image-smoothing.ipynb)
  * [10.3.2 2D Convolution for edge detection](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.3.2-2dconv-edge-detection.ipynb)
  * [10.4.1 3D Convolution for motion detection](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.4.1-3dconv-motion-detection.ipynb)
  * [10.5.3 Transpose Convolution for upsampling](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.5.3-transpose-conv-upsampling.ipynb)

* Chapter 11
  * [11.2.1 Convolutional Neural Networks for Image Classification - LeNet](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.2.1-mnist-lenet.ipynb)
  * [11.3.1 VGG (Visual Geometry Group) Net](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.3.1-vgg.ipynb)
  * [11.3.2 Inception: Network in Network paradigm](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.3.2-inception.ipynb)
  * [11.3.3 ResNet](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.3.3-resnet.ipynb)
  * [11.3.4 PyTorch Lightning: LeNet](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.3.4-mnist-lenet-lightning.ipynb)
  * [11.3.4 PyTorch Lightning: Deep CNN Classifier](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.3.4-hymenoptera-deep-classifier.ipynb)
  * [11.5 Faster R-CNN](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch11/11.5-frcnn.ipynb)

* Chapter 13
  * [13.2 Bayesian Inferencing for mean in Gaussian likelihood, known variance](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.2-bayesian-inference-unknown-mean.ipynb)
  * [13.5.1 Gamma Distribution](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.5.1-gamma-distribution.ipynb)
  * [13.5 Bayesian Inferencing of Precision of Gaussian likelihood, known Mean](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.5-bayesian-inference-unknown-variance.ipynb)
  * [13.6 Bayesian Inferencing of both Mean and Precision of Gaussian likelihood](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.6-bayesian-inference-unknown-mean-variance.ipynb)
  * [13.7 Statsville Revisited: Bayesian inferencing to predict if a Statsville resident is female based on height](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.7-statsville-revisited.ipynb)
  * [13.8 Multivariate Bayesian Inferencing of Mean of Gaussian likelihood, known Precision](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch13/13.8-bayesian-inference-unknown-mean-multivariate.ipynb)
 
* Chapter 14
  * [14.1 Principal Components Analysis (PCA) recap](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch14/14.1-pca.ipynb)
  * [14.2 Auto Encoders](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch14/14.2-auto-encoders.ipynb)
  * [14.4 Variational Auto Encoders](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch14/14.4-variational-auto-encoders.ipynb)
