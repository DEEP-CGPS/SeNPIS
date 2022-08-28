# SeNPIS

Sequential Network Pruning by class-wise Importance Score

SeNPIS is a pruning method for compressing sequential CNNs.
It eliminates non-relevant filters or neurons based on an class-wise importance score.
It provides a better relationship between accuracy, FLOPs and pruned model parameters than its competitors (Weight, Taylor, Gradient, and LRP).

If you find this code useful in your research, please consider citing:

@article{PACHON2022109558,
title = {SeNPIS: Sequential Network Pruning by class-wise Importance Score},
journal = {Applied Soft Computing},
pages = {109558},
year = {2022},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2022.109558},
url = {https://www.sciencedirect.com/science/article/pii/S1568494622006238},
author = {César G. Pachón and Dora M. Ballesteros and Diego Renza},
}

In addition, in case you need the models already trained in pytorch of VGG16 and AlexNet, for Cifar-10 and Scene-15, with different pruning values (10, 30, 50, 70, 90)% Pn and Pn+FT presented in the paper you can download them from: https://drive.google.com/drive/folders/1ZBAE761KATOhjnI0tFV70LYs_tZE_-ku?usp=sharing
