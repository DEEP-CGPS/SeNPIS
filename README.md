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
keywords = {Deep learning, Model compression, Pruning algorithm, Importance score, Convolutional neural network},
abstract = {In the last decade, pattern recognition and decision making from images has mainly focused on the development of deep learning architectures, with different types of networks such as sequential, residual and parallel. Although the depth and size varies between models, they all have in common that they can contain multiple filters or neurons that are not important for the purpose of prediction, and that do negatively impact the size of the model and their inference times. Therefore, it is advantageous to use pruning methods that, while largely maintaining the initial performance of the classifier, significantly reduce its size and FLOPs. In parameter reduction, the decision rule is generally based on mathematical criteria, e.g. the amplitude of the weights, but not on the actual impact of the filter or neuron on the classifier performance for each of the classes. Therefore, we propose SeNPIS as a method that involves both filter and neuron selection based on a class-wise importance score, and network resizing to increase parameter reduction and FLOPs in sequential CNN networks. Several tests were performed to compare SeNPIS with other representative state-of-the-art methods, for the CIFAR-10 and Scene-15 datasets. It was found that for similar values of accuracy, and even in some cases with a slight increase in accuracy, SeNPIS significantly reduces the number of parameters by up to an additional 23.5% (i.e., a 51.05% reduction with SeNPIS versus a 27.53% reduction with Gradient) and FLOPs by up to an additional 26.6% (i.e., a 74.82% reduction with SeNPIS versus a 48.16% reduction with Weight) compared to the Weight, Taylor, Gradient and LRP methods.}
}

In addition, in case you need the models already trained in pytorch of VGG16 and AlexNet, for Cifar-10 and Scene-15, with different pruning values (10, 30, 50, 70, 90)% Pn and Pn+FT presented in the paper you can download them from: https://drive.google.com/drive/folders/1ZBAE761KATOhjnI0tFV70LYs_tZE_-ku?usp=sharing
