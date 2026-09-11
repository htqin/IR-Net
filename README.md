# IR-Net: Forward and Backward Information Retention for Accurate Binary Neural Networks

**CVPR 2020**

Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen, Ziran Wei, Fengwei Yu, Jingkuan Song

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_Networks_CVPR_2020_paper.html) | [arXiv](https://arxiv.org/abs/1909.10788) | [Citation](#citation)

**IR-Net trains binary neural networks while retaining information in both forward activations and backward gradients.** Libra Parameter Binarization (Libra-PB) balances and standardizes weights; the Error Decay Estimator (EDE) progressively adjusts the gradient approximation during training.

## Results and evidence

The following are published results, not new runs. W/A denotes weight/activation precision; first and last layers remain full precision. ImageNet 1/1 results use the Bi-Real ResNet structure; 1/32 results use the normal structure (Table 4).

| ImageNet model | Method | W/A | Top-1 (%) | Top-5 (%) |
| --- | --- | --- | --- | --- |
| ResNet-18 | Full precision | 32/32 | 69.6 | 89.2 |
| ResNet-18 | Bi-Real | 1/1 | 56.4 | 79.5 |
| ResNet-18 | IR-Net | 1/1 | 58.1 | 80.0 |
| ResNet-18 | IR-Net | 1/32 | 66.5 | 86.8 |
| ResNet-34 | Bi-Real | 1/1 | 62.2 | 83.9 |
| ResNet-34 | IR-Net | 1/1 | 62.9 | 84.1 |
| ResNet-34 | IR-Net | 1/32 | 70.4 | 89.5 |

### What this paper supports

- **Forward information retention:** Libra-PB reduces binarization information loss through balanced, standardized weights and bit-shift scaling (Section 4.1; Figure 5).
- **Backward information retention:** EDE trades off updating ability early in training and gradient approximation accuracy later (Section 4.2; Figure 6).
- **Complementary components:** CIFAR-10 ResNet-20 improves from 83.8% for vanilla binarization to 84.9% with Libra-PB, 85.2% with EDE, and 86.5% with both (Table 2).
- **ImageNet accuracy:** 1/1 ResNet-18 improves by 1.7 percentage points over Bi-Real under the comparison in Table 4; this does not imply parity with full precision.
- **Measured edge inference:** single-thread ResNet-18 on Raspberry Pi 3B (Cortex-A53, 1.2 GHz) takes 261.98 ms for IR-Net versus 1418.94 ms for FP32, approximately 5.4× faster (Section 5.3; Table 5). This is hardware latency, not a FLOPs ratio or a GPU speed claim.

## Implementation and reproduction

This repository provides PyTorch training/model code in [CIFAR-10](CIFAR-10) and [ImageNet](ImageNet). Binarization is learned with training data; IR-Net is not a data-free post-training quantizer. The original experiments use random initialization, SGD, and Hardtanh activations.

The reference environment is Python 3.6 and PyTorch 0.4.1. CIFAR-10 training used one NVIDIA GeForce GTX 1080 Ti; ImageNet training used 32. See the architecture-specific code and configurations before adapting the environment.

### CIFAR-10 results

Published accuracy from Table 3. The 86.5% ResNet-20 row uses the Bi-Real structure; the paper also reports 85.4% with the normal structure.

| Topology | W/A | Accuracy (%) |
| --- | --- | --- |
| ResNet-20 | 1/1 | 86.5 |
| ResNet-20 | 1/32 | 90.8 |
| VGG-Small | 1/1 | 90.4 |
| ResNet-18 | 1/1 | 91.5 |

## Citation

Please cite the published paper below. Open paper versions are linked at the top of this README.

```bibtex
@inproceedings{Qin:cvpr20,
  title = {Forward and Backward Information Retention for Accurate Binary Neural Networks},
  author = {Haotong Qin and Ruihao Gong and Xianglong Liu and Mingzhu Shen and Ziran Wei and Fengwei Yu and Jingkuan Song},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2020},
  pages = {2250--2259},
  url = {https://openaccess.thecvf.com/content_CVPR_2020/html/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_Networks_CVPR_2020_paper.html}
}
```
