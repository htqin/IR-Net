# IR-Net

This project is the PyTorch implementation of our accepted CVPR 2020 paper : forward and backward information retention for accurate binary neural networks.

Bibtex:

```
@inproceedings{Qin:cvpr20,
  author    = {Haotong Qin and Ruihao Gong and Xianglong Liu and Mingzhu Shen and Ziran Wei and Fengwei Yu and Jingkuan Song},
  title     = {Forward and Backward Information Retention for Accurate Binary Neural Networks},
  booktitle = {IEEE CVPR},
  year      = {2020},
 }
```

**IR-Net:** We implement our IR-Net using Pytorch because of its high flexibility and powerful automatic differentiation mechanism. When constructing a binarized model, we simply replace the convolutional layers in the origin models with the binary convolutional layer binarized by our method.

**Network Structures:** We employ the widely-used network structures including VGG-Small, ResNet-20, ResNet-18 for CIFAR-10, and ResNet-18, ResNet-34 for ImageNet. To prove the versatility of our IR-Net, we evaluate it on both the normal structure and the Bi-Real structure of ResNet. All convolutional and fully-connected layers except the first and last one are binarized, and we select Hardtanh as our activation function instead of ReLU.

**Initialization:** Our IR-Net is trained from scratch (random initialization) without leveraging any pre-trained model. To evaluate our IR-Net on various network architectures, we mostly follow the hyper-parameter settings of their original papers. Among the experiments, we apply SGD as our optimization algorithm.

**Dependencies**

- Python 3.6
- Pytorch == 0.4.1

For the GPUs, we use a single NVIDIA GeForce 1080TI when training IR-Net on the CIFAR-10 dataset and 32 NVIDIA GeForce 1080TI when training IR-Net on the ImageNet dataset.

**Accuracy:** 

​	CIFAR-10:

| Topology  | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-20 | 1 / 1           | 86.5         |
| ResNet-20 | 1 / 32          | 90.8         |
| VGG-Small | 1 / 1           | 90.4         |
| ResNet-18 | 1 / 1           | 91.5         |

​	ImageNet:

| Topology  | Bit-Width (W/A) | Top-1 (%) | Top-5 (%) |
| --------- | --------------- | --------- | --------- |
| ResNet-18 | 1 / 1           | 58.1      | 80.0      |
| ResNet-18 | 1 / 32          | 66.5      | 86.8      |
| ResNet-34 | 1 / 1           | 62.9      | 84.1      |
| ResNet-34 | 1 / 32          | 70.4      | 89.5      |

