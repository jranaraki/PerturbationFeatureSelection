---
title: "PerturbationFeatureSelection"
authors: "Javad Rahimipour Anaraki and Hamid Usefi"
date: '29/05/18'
---

## Notice
The code `PFS.m` is available now and a link to the paper will be provided soon. If you need more details and explanation about the algorithm, please contact [Javad Rahimipour Anaraki](http://www.cs.mun.ca/~jra066/) or [Hamid Usefi](http://www.math.mun.ca/~usefi/).

## Use case
To determine the most important features using the algorithm described in "A Feature Selection based on Perturbation Theory" by Javad Rahimipour Anaraki and Hamid Usefi

Here is two links to the paper: [arXiv](https://arxiv.org/abs/1902.09938) and [Expert Systems With Applications]()

## Compile
This code can be run using MATLAB R2006a and above

## Run
To run the code, open `PFS.m` and choose a dataset to apply the method to. The code strats reading the selected dataset using `readLargeCSV.m` written by [Cedric Wannaz](https://www.mathworks.com/matlabcentral/profile/authors/1078046-cedric-wannaz). Then it selects the most important features and find the best subset by looking at the classification accuracies returned by `cAcc.m` divided by the size of seleted subsets. Finally, a subset with the best accuracy and the smallest number of features is selected and returned as the output. All datasets are stored in *Data* folder and originally adopted from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and [ASU feature selection datasets](http://featureselection.asu.edu/)

## Note
 - In order to get accuracy using decision tree, support vector machine or k-nearest neighbour the corresponding line in the `cAccInner.m` should be uncommented
 - Datasets should have no column and/or row names, and the class values should be all numeric
