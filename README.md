Metastatic Cancer Detection in Histopathology Images
====================================================

**Problem Description:**

The challenge is a binary image classification task aimed at detecting metastatic cancer in small image patches extracted from larger digital pathology scans. Specifically, the goal is to develop an algorithm that can accurately classify these patches as either containing metastatic cancer (positive label) or not (negative label). The presence of cancer is determined solely by the presence of tumor tissue within the center 32x32 pixel region of each patch. This task has significant clinical relevance, as it directly addresses the critical issue of cancer metastasis detection.

**Data Description:**

The dataset is a modified version of the PatchCamelyon (PCam) benchmark dataset. It consists of a large number of small, color pathology images. Will Cukierski. Histopathologic Cancer Detection. <https://kaggle.com/competitions/histopathologic-cancer-detection>, 2018. Kaggle.

-   **Size:** The dataset comprises thousands of image patches, split into training and testing sets.
-   **Dimension:** Each image patch has a fixed dimension of 96x96 pixels with 3 color channels (RGB).
-   **Structure:**
    -   The data (which is all stored in `data/`) is organized into two subdirectories: `train/` and `test/`.
    -   The `train` folder contains images for training the model, and the `test` folder contains images for which predictions are to be made.
    -   A `train_labels.csv` file provides the ground truth labels for the training images, mapping image IDs to binary labels (0 or 1).
    -   The labels are only determined by the center 32x32 pixel region of each image.
    -   The outer area of the 96x96 images are provided to support fully convolutional network architectures.
-   **Format:** Images are in standard image formats (tif).
-   **Class Imbalance:** It's important to note that the dataset exhibits class imbalance, with an unequal distribution of positive and negative samples.
-   **No Duplicates:** The Kaggle version of the PCam dataset has been processed to remove duplicate images, ensuring a cleaner training set.

Technical Highlights
--------------------

### 🔍 Exploratory Data Analysis (EDA)

-   **Center Region Visualization:** Highlighted diagnostic 32x32px regions in sample images

-   **Color Analysis:** Compared RGB distributions between classes in critical regions

-   **Class Weights:** Calculated loss weights (Negative: 0.84, Positive: 1.23) to address imbalance

### 🛠 Technical Implementation

**Data Pipeline**

-   TensorFlow `tf.data` for efficient loading

-   On-the-fly augmentation: Horizontal flips, brightness/contrast adjustments

-   Stratified 80/20 train-validation split

**Model Development**\
*Two architectures compared via Keras Tuner:*

| Architecture | Val Acc | Params | Inference Speed | Key Feature |
| --- | --- | --- | --- | --- |
| Custom CNN | 96.4% | 1.28M | 74ms/batch | Center-region optimized |
| ResNet50 | 89.6% | 23.5M | 112ms/batch | Transfer learning |

### Hyperparameter Tuning Approach

**Optimization Protocol:**

1.  **Search Space:**

    hp = {\
    'architecture': Choice(['simple_cnn', 'ResNet50']),\
    'conv_filters': Int(16, 64, step=16),\
    'learning_rate': Float(1e-4, 1e-2, log=True),\
    'dense_units': Int(64, 256)\
    }

2.  **Search Strategy:**

    -   10 RandomSearch trials

    -   Early stopping (patience=3)

    -   Batch size fixed at 64

3.  **Resource Allocation:**

    -   10 epochs per trial

    -   20% validation split

    -   Class-weighted loss
  
  ### 📈 Training Strategy

-   **Class-weighted loss** for imbalance mitigation

-   **Dynamic LR reduction** (ReduceLROnPlateau)

-   **Early stopping** with ModelCheckpoint

Key Innovations
---------------

1.  **Diagnostic Region Prioritization**

    -   Custom CNN architecture designed for small tumor regions

    -   GlobalAveragePooling instead of Flatten for spatial awareness

2.  **Resource-Efficient Design**

    -   1.28M parameter model outperformed 23.5M parameter ResNet50

    -   Optimized for CPU training constraints

Results
-------

| Metric | Value |
| --- | --- |
| Val Accuracy | 96.4% |
| Test Accuracy | ~87% |

Future Directions
-----------------

1.  **Architecture Improvements**

    -   Implement attention mechanisms for region focus

    -   Test EfficientNet/ViT architectures

2.  **Advanced Training**

    -   Unfreeze & fine-tune ResNet50 layers

    -   Experiment with focal loss

3.  **Data Enhancements**

    -   Add rotation/elastic transforms

    -   Test stain normalization techniques
