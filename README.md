# SeNPIS

Sequential Network Pruning by class-wise Importance Score

SeNPIS is a pruning method for compressing sequential CNNs.
It eliminates non-relevant filters or neurons based on an class-wise importance score.
It provides a better relationship between accuracy, FLOPs and pruned model parameters than its competitors (Weight, Taylor, Gradient, and LRP).

If you find this code useful in your research, please consider citing:

@article{..........,
  title={...........},
  author={...........},
  journal={............},
  pages={............},
  year={2021},
  publisher={..........}
}

In addition, in case you need the models already trained in pytorch of VGG16 and AlexNet, for Cifar-10 and Scene-15, with different pruning values (10, 30, 50, 70, 90)% Pn and Pn+FT presented in the paper you can download them from: ...
