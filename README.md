# ImageRetrieval
## Installation

To use the ImageRetrieval package, you can follow these installation steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```


2. Install the `ImageRetrieval` package itself:

    ```bash
    pip install .
    ```

    This will automatically install packages

3. You're all set! Now lets look at results from our approach

* Resnet50 Inference (Frozen Layers): The initial model achieved a ROC-AUC of 0.75.
* Resnet50 with CenterLoss + CrossEntropy: Introducing the CenterLoss and CrossEntropy loss functions along with a feature dimension of 512 led to an improvement, yielding a ROC-AUC of 0.815
* Resnet50 with CenterLoss + CrossEntropy + ArcFaceLoss: Incorporating ArcFaceLoss alongside the previous losses, and enabling augmentation, further improved the performance to a ROC-AUC of 0.84.
* Resnet50 with CenterLoss + CrossEntropy + ArcFaceLoss (FeatureDim=4096): Increasing the feature dimension to 4096 while retaining augmentation boosted the ROC-AUC to 0.85.
* Resnet101 with CenterLoss + CrossEntropy + ArcFaceLoss (FeatureDim=4096): Utilizing a larger Resnet101 model while maintaining the augmented feature dimension resulted in a ROC-AUC of 0.87.
* Hybrid CNN-Transformer (Resnet101 + SwinTransformer): Creating a hybrid model by combining a Resnet101 CNN with a SwinTransformer, and augmenting the feature dimension to 8192, achieved a remarkable ROC-AUC of 0.9. The integration of the SwinTransformer as a helper for CNN feature extraction played a pivotal role in achieving this state-of-the-art result.

