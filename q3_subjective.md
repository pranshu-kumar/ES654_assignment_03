# ES654: Machine Learning | Assignment 03
### Pranshu Kumar Gond (18110124)
---
## Question 03

### c) Overall Accuracy and Confusion Matrix
The decrease in the loss function is presented in the given graph, 

![](mcr_loss.png)

The overall accuracy of the model came out to be: **0.8777**

The Confusion Matrix is given below,

![](confusionmatrix.png)

Looking closely at the confusion matrix, 

The two digits that got the most confused were: digit **1** and **9**

The digits that were easily predicted are: **0**, **6** and **7**

### d) PCA Plot
After doing PCA on the given dataset, the dimensionality of the dataset was reduced to **2**. 

The PCA plot is given below, 

![](pcaplot.png)

From the plot, we can infer the following:
* The digits are correctly classified as different clusters. However, some clusters partially overlap. This suggests that these digits are difficult to seperate. 
* The digit **0** is at the top seperated from all the clusters suggesting that it is easy to distinguish 0 (The confusion matrix suggests the same). The same goes for the digit **6**
