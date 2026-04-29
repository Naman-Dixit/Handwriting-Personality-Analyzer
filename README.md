# Handwriting Personality Analyzer

**MLDL Capstone Project**
B.Tech Computer Science and Engineering
Author: Naman Dixit | Sap ID: 500125539 | Section: B7 Data Science |Year - 3, Semester - 6

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Visualizations](#visualizations)
6. [Future Scope](#future-scope)
7. [Tech Stack](#tech-stack)
8. [Limitations and Notes](#limitations-and-notes)

---

## Project Overview

This project implements an end-to-end machine learning pipeline to analyze handwriting patterns and predict personality traits. It combines unsupervised clustering with supervised regression to identify distinct handwriting styles and estimate personality scores.

The pipeline loads the MNIST handwritten digit dataset, extracts statistical and structural features from each image, groups samples into style clusters using KMeans, and trains a Random Forest Regressor to predict an Extraversion score as a proxy for personality.

---

## Objectives

- Load and preprocess handwritten image data from raw `.ubyte` binary format.
- Extract meaningful visual and statistical features from each image.
- Apply unsupervised clustering to group images by handwriting style.
- Train a regression model to predict personality trait scores.
- Evaluate model performance using R2, RMSE, and cross-validation.
- Visualize clustering structure, feature importance, and prediction quality.

---

## Methodology

### Dataset

The MNIST training split is used: 60,000 grayscale images of handwritten digits, each 28x28 pixels. Data is loaded directly from the raw `.ubyte` binary files using a custom NumPy parser, without relying on any pre-packaged dataset loaders.

```
Images shape: (60000, 28, 28)
Labels shape: (60000,)
```

### Feature Extraction

Seven features are extracted per image after normalizing pixel values to [0, 1]:

| Feature | Description |
|---|---|
| Mean | Average pixel intensity |
| Std | Standard deviation of pixel intensities |
| Max | Maximum pixel value |
| Min | Minimum pixel value |
| Stroke Density | Count of pixels exceeding threshold 0.3 |
| Center X | Mean X coordinate of pixels above threshold |
| Center Y | Mean Y coordinate of pixels above threshold |

```
Feature matrix shape: (60000, 7)
```

### Preprocessing

`StandardScaler` is applied to the feature matrix to zero-center and unit-normalize all features. This is necessary because KMeans is sensitive to feature scale.

### Clustering

`KMeans` with `n_clusters=4`, `random_state=42`, and `n_init=10` is fit on the scaled features. Cluster labels are appended to the feature matrix as an eighth input to the regression model.

PCA is used to project features to 2D for cluster visualization only and is not part of the modeling pipeline.

### Target Generation

Since no labeled handwriting-personality dataset exists, Extraversion scores are generated synthetically as a weighted combination of scaled features with added Gaussian noise:

```python
y_raw = (
    0.35 * X_scaled[:, 0] +   # Mean
    0.25 * X_scaled[:, 1] +   # Std
    0.15 * X_scaled[:, 4] +   # Stroke Density
    np.random.normal(0, 0.2, len(X_scaled))
)
```

Scores are then normalized to [0, 1] using `MinMaxScaler`.

### Regression Model

A `RandomForestRegressor` with `n_estimators=300`, `max_depth=10`, and `random_state=42` is trained on an 80/20 train-test split. Input is the 8-dimensional vector of scaled features plus cluster label.

### Inference

A helper function `analyze_handwriting(image)` accepts any 2D image, resizes it to 28x28, extracts features, scales them, predicts the cluster, and returns the Extraversion score. Sample output on `images[100]`:

```
Cluster            : 0
Extraversion Score : 0.268
Model R2           : 0.931
```

---

## Results

| Metric | Value |
|---|---|
| R2 Score (test set) | 0.9313 |
| RMSE (test set) | 0.0311 |
| CV Mean R2 (5-fold) | 0.9312 |

---

## Visualizations

The notebook produces the following plots:

- **PCA Cluster Scatter**: 2D projection of the feature space, color-coded by KMeans cluster assignment.
- **Feature Importance Bar Chart**: Ranks all 8 input features by their contribution to the Random Forest predictions using `model.feature_importances_`.

---

 
## Alternative Datasets
 
The following datasets can be used as direct replacements or supplements for MNIST in this pipeline:
 
| Dataset | Description |
|---|---|
| EMNIST | Extension of MNIST with handwritten letters and digits from 814,255 samples. A natural drop-in upgrade. |
| IAM Handwriting Database | 1,539 handwritten English text pages from 657 writers, with word and line level segmentation. |
| CVL Handwriting Database | Handwritten text from 310 writers in English and German. Useful for writer-style analysis. |
 
---

## Future Scope

- Replace synthetic labels with real handwriting-personality data (e.g., IAM database paired with Big Five assessments).
- Extend prediction to all five Big Five traits using multi-output regression.
- Replace handcrafted features with a CNN or Vision Transformer encoder trained end-to-end on raw images.
- Add SHAP explainability to interpret individual predictions.
- Explore online handwriting data (pen speed, stroke order, pressure) from stylus-enabled devices.
- Evaluate in a writer-independent setting where test writers are unseen during training.

---

## Tech Stack

- Python 3
- NumPy
- OpenCV (`cv2`)
- Scikit-learn (StandardScaler, KMeans, PCA, RandomForestRegressor, MinMaxScaler)
- Matplotlib
- Seaborn

---

## Limitations and Notes

Personality scores are synthetically generated and do not reflect real psychological measurements. The high R2 values are expected given that the target is a deterministic function of the input features with moderate noise. These results would be substantially lower on a real handwriting-personality dataset and should not be interpreted as evidence that handwriting predicts personality.

---

Developed as part of an MLDL Capstone Project, B.Tech CSE.
Naman Dixit | 500125539 | B7 Data Science
