# Cricket Shot Classification Using Pose Estimation and Deep Learning

A computer vision project that combines **human pose estimation, skeletal feature visualization, heatmap generation, and deep learning** to explore the classification of cricket shots. The project uses OpenPose to detect human body keypoints and evaluates CNN, DNN, and VGG16 architectures for four-class cricket shot classification.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Cricket Shot Categories](#cricket-shot-categories)
- [Deep Learning Models](#deep-learning-models)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Execution](#execution)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

## Overview

Analyzing cricket shots involves understanding player posture, body positioning, and movement patterns. Computer vision and deep learning provide techniques for extracting visual features and identifying patterns associated with different cricket shots.

This project implements an image-processing pipeline using OpenPose and a pretrained COCO body keypoint model to detect 18 human body keypoints. The detected keypoints are used to generate skeletal visualizations and heatmaps, which are saved as processed images for subsequent classification experiments.

Three neural network architectures are explored:

- **Convolutional Neural Network (CNN):** Learns spatial features from grayscale images.
- **Deep Neural Network (DNN):** Learns patterns from flattened image representations.
- **VGG16 with Transfer Learning:** Uses an ImageNet-pretrained convolutional backbone with custom classification layers.

The models are evaluated on four cricket shot categories, and their performance is examined using accuracy, classification reports, and confusion matrices.

## Objectives

- Detect human body keypoints in cricket images using OpenPose.
- Generate skeletal overlays to visualize player posture.
- Create heatmaps representing the spatial distribution of detected body joints.
- Prepare processed images for deep learning experiments.
- Implement and compare CNN, DNN, and VGG16 architectures.
- Evaluate classification performance across four cricket shot categories.
- Explore the application of pose estimation and deep learning in sports analytics.

## System Architecture

The system follows a computer vision preprocessing pipeline followed by multiple classification experiments.

### Architecture Diagram

```text
                   CRICKET IMAGE DATASET
                            |
                            v
                    IMAGE ACQUISITION
                            |
                            v
                   OPENPOSE MODEL
                (COCO Body Keypoints)
                            |
                            v
                  KEYPOINT EXTRACTION
                    (18 Keypoints)
                            |
                 +----------+----------+
                 |                     |
                 v                     v
        SKELETAL VISUALIZATION    HEATMAP GENERATION
                 |                     |
                 v                     v
          SKELETON IMAGES         HEATMAP IMAGES
                 |                     |
                 +----------+----------+
                            |
                            v
                    PROCESSED IMAGES
                            |
                            v
                    DATA PREPROCESSING
                 +----------+----------+
                 |                     |
                 v                     v
          GRAYSCALE IMAGES          RGB IMAGES
                 |                     |
                 |                     |
           +-----+-----+               |
           |           |               |
           v           v               v
          CNN         DNN             VGG16
           |           |               |
           v           v               v
      Convolutional  Fully         Pretrained
        Features   Connected       Feature Maps
                     Layers            |
           |           |               v
           |           |        Global Average Pooling
           |           |               |
           |           |               v
           |           |         Dense Classification
           |           |               |
           +-----------+---------------+
                            |
                            v
                  FOUR-CLASS PREDICTION
                            |
                            v
                    MODEL EVALUATION
                            |
                            v
                Accuracy and Classification
                  Reports and Confusion Matrices
```

**Architecture note:** The diagram summarizes the overall workflow. In the current notebook, CNN and DNN load grayscale images from the processed-image directories, while VGG16 loads images as RGB. The classification loaders do not explicitly filter filenames to separate skeleton images from heatmap images. Therefore, the diagram represents the overall processing and experimentation workflow rather than a strictly isolated, pose-only feature pipeline.

## Methodology

### 1. Dataset Organization

The source dataset contains images organized into four class-specific directories.

Each image belongs to one of the supported cricket shot categories.

### 2. Human Pose Estimation

The project uses OpenPose through OpenCV's Deep Neural Network module.

The pretrained model is loaded using:

- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`

The model produces confidence maps for 18 body keypoints.

For each keypoint, the implementation identifies the maximum-confidence location and retains it when the confidence exceeds `0.1`.

The detected coordinates are mapped back to the original image dimensions.

### 3. Skeletal Visualization

The detected body keypoints are connected using predefined skeletal connections.

The visualization includes:

- Circular markers for detected keypoints.
- Lines connecting related body joints.
- An overlay showing the player's estimated body structure.

The generated skeletal images provide a visual representation of player posture during cricket shots.

### 4. Heatmap Generation

The project generates a combined heatmap from the detected keypoints.

The procedure involves:

1. Creating a blank single-channel heatmap.
2. Marking each detected keypoint with a circular region.
3. Applying Gaussian blur to smooth the joint regions.
4. Combining the individual joint heatmaps.
5. Normalizing the resulting heatmap to an intensity range of 0–255.

The resulting heatmap emphasizes the spatial distribution of detected body joints.

### 5. Image Preprocessing

Processed images are prepared for the classification models using different input representations.

**CNN and DNN preprocessing:**

- Load processed images in grayscale.
- Resize images to 64 × 64 pixels.
- Normalize pixel values to the range 0–1.
- Convert class labels into numerical representations.
- Split the dataset into training and testing subsets.

**VGG16 preprocessing:**

- Load processed images in RGB format.
- Resize images to 100 × 100 pixels.
- Normalize pixel values to the range 0–1.
- Encode class labels using one-hot encoding.
- Split the dataset into training and testing subsets.
- Apply image augmentation to the training data.

### 6. Model Training

Three neural network architectures are implemented and trained independently.

The models use the same four-class classification objective but differ in their input representations and feature-learning approaches.

### 7. Model Evaluation

The trained models are evaluated using held-out test data.

The evaluation includes accuracy, confusion matrices, and classification reports containing precision, recall, and F1-score.

## Cricket Shot Categories

The classification task includes four categories.

| Category | Description |
|---|---|
| Drive | An attacking shot generally played with a straight or angled bat. |
| Leg Glance / Flick | A shot directed toward the leg side using bat angle and wrist movement. |
| Pull Shot | A horizontal-bat shot commonly played against a short-pitched delivery. |
| Sweep | A shot generally played toward the leg side against a slower or shorter delivery. |

The dataset directory names used in the notebook are:

```text
drive/
legglance-flick/
pullshot/
sweep/
```

## Deep Learning Models

### 1. Convolutional Neural Network (CNN)

The CNN learns spatial patterns through convolutional and max-pooling layers.

**Architecture:**

```text
Input Image: 64 × 64 × 1
          |
          v
Conv2D: 32 Filters, 3 × 3
Activation: ReLU
          |
          v
MaxPooling2D: 2 × 2
          |
          v
Conv2D: 64 Filters, 3 × 3
Activation: ReLU
          |
          v
MaxPooling2D: 2 × 2
          |
          v
Flatten
          |
          v
Dense: 64 Neurons
Activation: ReLU
          |
          v
Dense: 4 Neurons
Activation: Softmax
          |
          v
Predicted Cricket Shot
```

**Configuration:**

| Parameter | Value |
|---|---|
| Input shape | 64 × 64 × 1 |
| First convolution | 32 filters, 3 × 3 |
| Second convolution | 64 filters, 3 × 3 |
| Pooling | 2 × 2 max pooling |
| Hidden dense layer | 64 neurons |
| Output layer | 4 neurons |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Training epochs | 20 |
| Validation split | 20% of training data |

The final softmax layer produces class probabilities for the four cricket shot categories.

### 2. Deep Neural Network (DNN)

The DNN operates on flattened grayscale image representations.

Each 64 × 64 image is converted into a vector containing 4,096 pixel values.

**Architecture:**

```text
Input: 4096 Features
          |
          v
Dense: 512 Neurons
Activation: ReLU
          |
          v
Dropout: 0.3
          |
          v
Dense: 256 Neurons
Activation: ReLU
          |
          v
Dropout: 0.3
          |
          v
Dense: 128 Neurons
Activation: ReLU
          |
          v
Dropout: 0.2
          |
          v
Dense: 4 Neurons
Activation: Softmax
          |
          v
Predicted Cricket Shot
```

**Configuration:**

| Parameter | Value |
|---|---|
| Input features | 4,096 |
| Hidden layer 1 | 512 neurons |
| Hidden layer 2 | 256 neurons |
| Hidden layer 3 | 128 neurons |
| Dropout rates | 0.3, 0.3, 0.2 |
| Output layer | 4 neurons |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Training epochs | 60 |
| Validation split | 20% of training data |

Dropout layers are used to reduce overfitting by randomly deactivating a proportion of neurons during training.

### 3. VGG16 Transfer Learning

VGG16 is a convolutional neural network pretrained on ImageNet.

The project uses the pretrained convolutional base as a feature extractor and adds custom classification layers for the cricket shot classification task.

**Architecture:**

```text
Input Image: 100 × 100 × 3
          |
          v
VGG16 Backbone
ImageNet Pretrained Weights
          |
          v
Frozen Convolutional Layers
          |
          v
Global Average Pooling
          |
          v
Dense: 1024 Neurons
Activation: ReLU
          |
          v
Dense: 4 Neurons
Activation: Softmax
          |
          v
Predicted Cricket Shot
```

**Configuration:**

| Parameter | Value |
|---|---|
| Base architecture | VGG16 |
| Pretrained weights | ImageNet |
| Input shape | 100 × 100 × 3 |
| Include original classification head | No |
| Trainable backbone layers | Frozen |
| Pooling | Global average pooling |
| Custom hidden layer | 1,024 neurons |
| Output layer | 4 neurons |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Categorical cross-entropy |
| Training epochs | 40 |
| Batch size | 32 |

The training pipeline also defines image augmentation with rotation, width and height shifts, shear, zoom, and horizontal flipping.

**Implementation detail:** Although the augmentation generators are created in the notebook, the VGG16 model is trained using `model.fit(X_train, y_train, ...)` rather than the augmented training generator. Consequently, the configured augmentations are not actually applied during the current VGG16 training run.

### Additional Experiment: Random Forest

The notebook imports `RandomForestClassifier` and prepares a feature matrix from resized, flattened grayscale images.

However, the provided implementation does not train or evaluate the Random Forest classifier. It should therefore be considered an additional experiment that has been started, rather than a completed classification model.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| OpenCV | Image processing and DNN-based pose estimation |
| OpenPose | Human body keypoint detection |
| COCO Body Model | Pretrained body keypoint representation |
| TensorFlow | Deep learning framework |
| Keras | CNN, DNN, and VGG16 model implementation |
| NumPy | Numerical operations and array manipulation |
| Scikit-learn | Dataset splitting, label encoding, and evaluation metrics |
| Matplotlib | Visualization of model results |
| Seaborn | Confusion matrix visualization |
| Jupyter Notebook | Interactive development and experimentation |

## Project Structure

The repository can be organized as follows:

```text
cricket-shot-classification/
│
├── Pose_Estimation_In_Cricket.ipynb
│
├── README.md
│
├── dataset/
│   ├── drive/
│   ├── legglance-flick/
│   ├── pullshot/
│   └── sweep/
│
├── processed_images/
│   ├── drive/
│   ├── legglance-flick/
│   ├── pullshot/
│   └── sweep/
│
└── models/
    ├── cnn_model.h5
    ├── dnn_model.h5
    └── vgg16_model.h5
```

**Note:** This is a suggested repository layout. The notebook saves processed images and model files to configured output paths, but the complete dataset, generated images, and saved models are not necessarily included in the repository.

## Installation

### Prerequisites

- Python 3.9 or a compatible Python environment.
- Jupyter Notebook or Google Colab.
- A cricket image dataset organized into four classes.
- The pretrained OpenPose Caffe model files.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd cricket-shot-classification
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow jupyter
```

### 3. Download the OpenPose Model Files

The pose estimation component requires:

- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`

Obtain the compatible pretrained model files from the official OpenPose project.

Store the files locally and update the paths in the notebook:

```python
proto_file = "path/to/pose_deploy_linevec.prototxt"
weights_file = "path/to/pose_iter_440000.caffemodel"
```

The paths must point to existing files before the OpenPose network can be initialized.

## Dataset Preparation

Organize the source dataset using the following structure:

```text
pose_estimation/
├── drive/
│   ├── image_001.jpg
│   └── image_002.jpg
│
├── legglance-flick/
│   ├── image_001.jpg
│   └── image_002.jpg
│
├── pullshot/
│   ├── image_001.jpg
│   └── image_002.jpg
│
└── sweep/
    ├── image_001.jpg
    └── image_002.jpg
```

Supported source image extensions in the pose preprocessing loop include:

- `.jpg`
- `.jpeg`
- `.png`

Update the dataset path in the notebook:

```python
data_dir = "path/to/pose_estimation"
```

The notebook generates an output directory named `processed_images`, containing skeletal overlays and heatmaps for the source images.

## Execution

### Step 1: Configure the Paths

Update the following variables to match your environment:

```python
proto_file = "path/to/pose_deploy_linevec.prototxt"
weights_file = "path/to/pose_iter_440000.caffemodel"
data_dir = "path/to/pose_estimation"
```

### Step 2: Generate Skeletal Images and Heatmaps

Run the pose estimation and image preprocessing cells.

The generated images are saved under:

```text
processed_images/
├── drive/
├── legglance-flick/
├── pullshot/
└── sweep/
```

Each class directory contains skeletal overlays and heatmap images generated from its source images.

### Step 3: Train the CNN

Execute the CNN cells to:

- Prepare grayscale images.
- Encode class labels.
- Split the dataset.
- Train the CNN.
- Evaluate the model.
- Generate a confusion matrix and classification report.

### Step 4: Train the DNN

Execute the DNN cells to flatten the image representations, train the fully connected neural network, and evaluate its classification performance.

### Step 5: Train VGG16

Execute the VGG16 cells to load the RGB images, initialize the pretrained model, add the custom classification layers, and train the model.

### Step 6: Compare Model Performance

Run the evaluation and visualization cells to inspect classification reports, confusion matrices, and the model accuracy comparison plot.

### Important Execution Notes

- The notebook contains Windows-specific file paths that must be updated.
- The source dataset must exist before preprocessing begins.
- The OpenPose model files must be accessible.
- The classification loaders currently read all matching images within each processed class directory, including generated skeleton and heatmap images.
- Run the notebook cells in the intended order to ensure that variables, datasets, and trained models are initialized.
- The CNN, DNN, and VGG16 experiments use different preprocessing configurations, so their results should be interpreted with that difference in mind.

## Evaluation Metrics

The project uses several metrics to assess classification performance.

### Accuracy

Measures the proportion of correctly classified samples among all evaluated samples.

### Precision

Measures the proportion of predicted positive samples for a class that are correctly classified.

### Recall

Measures the proportion of actual samples belonging to a class that are correctly identified.

### F1-Score

The harmonic mean of precision and recall, useful for examining the balance between the two metrics.

### Confusion Matrix

Shows the distribution of correct and incorrect predictions across the four cricket shot categories.

These metrics provide a more informative assessment than accuracy alone, particularly when the dataset contains different numbers of images per class.

## Results

The notebook contains an accuracy comparison visualization with the following values:

| Model | Accuracy Value in Comparison Plot |
|---|---:|
| CNN | 97% |
| DNN | 76% |
| VGG16 | 84% |

**Important:** These values are hardcoded in the notebook's plotting cell. They are not automatically derived from the evaluation results, so they should not be treated as verified experimental measurements without confirmation.

For a reliable comparison, record the actual test accuracy returned by each model's evaluation step and use those measured values to generate the comparison plot.

The notebook also generates confusion matrices and classification reports for examining class-wise performance.

## Limitations

- **Image-based classification:** The current implementation classifies processed images rather than analyzing complete cricket video sequences.
- **Pose detection reliability:** Occlusion, image quality, and player orientation may affect keypoint detection.
- **Feature representation:** The classification loaders do not explicitly distinguish skeletal overlays from heatmaps.
- **Evaluation consistency:** The CNN, DNN, and VGG16 pipelines use different input representations and preprocessing configurations.
- **Augmentation:** The VGG16 augmentation generators are defined but are not used in the model's training call.
- **Random Forest:** The notebook prepares features but does not complete Random Forest training and evaluation.
- **Performance reporting:** The displayed accuracy comparison uses hardcoded values that require validation.
- **Reproducibility:** Local paths and external model files must be configured before the notebook can run in another environment.

## Future Enhancements

- Develop a dedicated pipeline for classifying skeletal keypoint representations.
- Compare raw images, skeletal overlays, and heatmaps as separate model inputs.
- Explore sequence-based models for cricket video analysis.
- Incorporate temporal information to distinguish shots with similar static postures.
- Evaluate additional architectures, including ResNet and MobileNet.
- Apply systematic hyperparameter tuning and cross-validation.
- Improve reproducibility through configuration files and standardized data splits.
- Implement the Random Forest experiment and compare its performance with neural network models.
- Integrate inference into a simple application for uploading images and predicting cricket shot categories.
- Validate performance on previously unseen images and videos.

## Conclusion

This project explores the application of computer vision and deep learning to cricket shot classification.

By combining OpenPose-based body keypoint extraction, skeletal visualization, heatmap generation, and multiple neural network architectures, it establishes a foundation for studying visual representations of cricket shots.

The CNN, DNN, and VGG16 implementations provide different approaches to image-based classification, while the evaluation pipeline supports further experimentation and comparison.

The project can be extended toward more robust sports analytics systems through improved dataset management, controlled experiments, keypoint-based classification, and video-level action recognition.

## Author

**Priyadharsini M.**

B.Tech in Computer Science and Engineering  
Specialization: Artificial Intelligence and Data Analytics
