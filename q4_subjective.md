# ES654: Machine Learning | Assignment 03
### Pranshu Kumar Gond (18110124)
---
## Question 04

Let, 
* **m:** Number of features in the dataset
* **c:** Number of classes
* **n:** Number of samples
* **e:** Number of iterations

### 1. Time Complexity of Logestic Regression

#### a) Training


The time complexity for training a Logistic Regression Model is ***O((m+1)cne)***
 
#### b) Prediction
The time complexity for predicting a set of samples is ***O((m+1)cn)***

### 2. Space Complexity of Logestic Regression

#### a) Training

During training we need, 

The matrix X which takes space ***O(nm)***

The matrix y which takes space ***O(n)***

The weight matrix which takes space ***O(m)***

Hence overall space complexity during training: ***O(mn + n + m)***

#### a) Prediction

During prediction we only need the final weight matrix which is of the order,
***O(m)***