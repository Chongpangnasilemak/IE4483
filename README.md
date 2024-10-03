# IE4483 Artificial Intelligence and Data Mining (Option 2)

## Project Overview

You are required to write an algorithm to classify whether an image contains either a dog or a cat. While this is easy for humans, it is more difficult for a computer. The project involves the following steps:

### 1. Loading and Preparing the Data

The dataset used in this project is based on, but not the same as, the official dataset from Kaggle's "Dogs vs. Cats" competition. You can download the training, validation, and test datasets from the provided Google Drive link. The datasets are organized as follows:


In the training and validation sets, there are two directories, `dog` and `cat`, which represent the corresponding image labels. The dataset includes:

- 10,000 images per class in the training set
- 2,500 images per class in the validation set
- 500 images in the test set that need classification

You can decide how many images to use for training and validation based on your computing resources. At least 2,000 images for training and 500 for validation are recommended.

### 2. Data Processing

You will need to pre-process your input images. Image augmentation (e.g., scaling, rotation, and flipping) is encouraged to improve classification performance.

**Hint:** You may downscale your input images if your machine cannot support heavy computation.

### 3. Model Selection

Design and develop a classification model to classify dog and cat images. You are encouraged to use neural networks for this task (though conventional classifiers can be used as a comparison). For example, you can use:

- A newly trained Convolutional Neural Network (CNN)
- A pretrained feature extraction backbone such as VGG or ResNet, and pass the features to a classifier layer

### 4. Model Training

Experiment with different model parameters, such as weights, initialization, and learning rates, to obtain optimal classification results. During training, use only the training set for optimization and the validation set for evaluation.

### 5. Prediction

Once the learning algorithm is set up, use the trained model to classify the dog and cat images in the test set. The results should be reported in the file `submission.csv` with two columns:

- `ID`: The ID of the images (refer to the sample `submission.csv`)
- `Predicted label`: (1 for dog, 0 for cat)

## Report Requirements

Your report should address the following questions:

### a) Data Usage and Preprocessing (10%)

- State the amount of image data used for training and testing.
- Describe the data preprocessing procedures and image augmentations (if any).

### b) Model Selection and Description (20%)

- Build and describe at least one machine learning model (e.g., CNN + linear layer).
- Provide a figure of the model architecture and explain the input/output dimensions, model structure, loss functions, and training strategy.
- Include your code and instructions for running it. If using a non-deterministic method, ensure reproducibility through averaging results, cross-validation, fixing random seed, etc.

### c) Model Parameters (5%)

- Discuss how you determined the parameters/settings (e.g., learning rate) and explain your rationale.

### d) Classification Accuracy and Submission (20%)

- Report the classification accuracy on the validation set.
- Apply the classifier to the test set and submit the `submission.csv` file with your results.

### e) Model Analysis (10%)

- Analyze some correctly and incorrectly classified samples from the test set.
- Select 1-2 cases and discuss the strengths and weaknesses of your model.

### f) Model and Data Processing Choices (10%)

- Discuss how different model and data processing choices may affect accuracy on the validation set.

### g) Multi-Category Image Classification (CIFAR-10) (15%)

- Apply and improve your classification algorithm for a multi-category image classification problem using the CIFAR-10 dataset.
- Describe the dataset, classification problem, and any changes you made compared to solving the Dogs vs. Cats problem.
- Report your results for the CIFAR-10 testing set.

### h) Data Imbalance in CIFAR-10 (10%)

- Train the classifier while some classes in the CIFAR-10 training dataset contain fewer labeled data.
- Explain at least two approaches you used to address the data imbalance issue and justify your choices.

## Notes

- If no meaningful results are obtained, describe what was done and attach relevant workings, code, or screenshots.
- Work in a group of three students and submit one report clearly indicating (1) group members and (2) respective contributions.
- Cite all references and sources used.
- Uphold the NTU Honour Code.
- Submit your report and the file `submission.csv` by **11:59pm, Friday, 17 Nov 2022** via NTULearn.
