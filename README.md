# DecisionTree-NaiveBayes-Implementation
## Decision Tree & Naïve Bayes using Python

### Project Description

This project implements two supervised machine learning classification algorithms:

* Decision Tree Classifier
* Naïve Bayes Classifier

The models are trained and evaluated using the Iris dataset from Scikit-learn. Their performance is compared using accuracy metrics and visualized using a bar graph.

---

## Aim

To implement and compare Decision Tree and Naïve Bayes classifiers using Python and evaluate their classification accuracy.

---

## Algorithms Used

### 1. Decision Tree

A tree-based supervised learning algorithm that splits data into subsets based on feature values.

Common splitting criteria include:

* Gini Index
* Entropy
* Information Gain

Decision Trees are easy to interpret and visualize, making them useful for classification problems.

---

### 2. Naïve Bayes

A probabilistic classifier based on Bayes' Theorem with the assumption that features are conditionally independent given the class label.

It is computationally efficient and performs well on many classification tasks, especially when the independence assumption approximately holds.

---

## Dataset

### Iris Dataset (from sklearn.datasets)

* Total Samples: 150
* Number of Classes: 3

  * Setosa
  * Versicolor
  * Virginica
* Number of Features: 4

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width

The dataset is commonly used for classification experiments and benchmarking machine learning algorithms.

---

## Technologies Used

* Python 3.x
* Scikit-learn
* Matplotlib
* NumPy

---

## Installation

Install the required dependencies using:

```
pip install numpy matplotlib scikit-learn
```
