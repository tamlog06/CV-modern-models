# R-CNN

This is a pytorch inplementation of R-CNN.

[論文へのリンク](https://arxiv.org/pdf/1311.2524.pdf) \
[参考サイト](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55)

## 要約

### Introduction
Precise recognition is important. Past works has made use of SIFT and HOG for visual recognition task.
However, visual recognition is much more hierarchical that just making use of SIFT and HOG doesn't give high accuracy. 
This paper shows that a CNN can lead to a higher object detection performance.
For this, we focus on two problems: localizing objects and training with only a small quantity of data.

## Object detection with R-CNN
R-CNN consists of three modules, category-independent region proposals, CNN, and SVM.

### Region proposals
We use selective search.


