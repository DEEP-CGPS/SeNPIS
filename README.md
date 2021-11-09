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

Ademas, en caso de requerir los modelos ya entrenados en pytorch de VGG16 y AlexNet, para Cifar-10 y Scene-15, con distintos
valores de poda (10,30,50,70,90)% Pn y Pn+FT presentados en el paper los pueden descargar de: 
