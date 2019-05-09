---
title: 'An Analysis of Support Vector Machines Learning Rules'
layout: post
date: 2019-05-09 22:48
image: "/assets/images/markdown.jpg"
tag:
- classification
category: blog
author: darsi-an
description: 
---


# Introduction 
Support Vector Machines (SVM) are a machine learning tool used to learn linear predictors in high dimensional feature space. The high dimensional feature space raises both sample complexity and computational complexity challenges. Consider the previous post on K-Nearest Neighbour, using Euclidean distance metric results in high misclassification error due to the curse of dimensionality. In this post, we explore a different machine learning method for classification in high dimensional feature space. This article will cover the different svm learning rules and introduce the concept of margins in relation to the regularized loss minimization paradigm. I will continue using digit recognition for examples but with different ML predictors. In this case I will use the three different SVM predictors to classify digits. 

# Margin
Consider a set $$ S = (x^1,y^1), ..., (x^n, y^n) $$ that is used to provide training instances of $$d$$ dimension for a classification model. A margin of a hyperplane is defined as the minimal distance between a point in the training dataset and the hyperplane. The true error is bounded by the margin over the training dataset and as a result the larger the margin the smaller the error, regardless of the Euclidean distance. In general, SVMs are more robust to perturbation and work well for high-dimensional data. 

Some of the different learning rules for SVM:
* Hard Margin 
* Soft Margin
* Kernelized SVM

## Hard-SVM
Hard margin SVM is the learning rule which determines the hyperplane with the maximum margin among all linear separators that separate the training dataset. Let the hyperplane be parameterized by $$w$$ where $$\|w\|_2$$ does not affect the margin and only the direction is important. Hence without loss of generality, we assume $$ \|w\|_2 = 1 $$.

> $$ \textbf{Rule:} $$  
$$ \underset{(w,b): \|w\|=1}{\arg\max}   \min \| {<w,x_i> +b} \| $$  
$$ s.t.  \forall i,  y_i(<w,x_i> + b) > 0 $$

An equivalent form of the rule as a quadratic optimization problem is defined as:

>  $$ \textbf{input:} (x_1,x_2),...,(x_m,y_m) $$  
$$ \textbf{solve:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ (w_0,b_0) = \underset{(w_0,b_0)}{\arg\min} = \|w\|^2$$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; s.t. &nbsp; $$  \forall i,  y_i(<w,x_i> + b) \geq 1 $$  
$$ \textbf{output:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ \hat{w} = \frac{w_0}{\|w_0\|} , \hat{b} = \frac{b_0}{\|w_0\|} $$

The above form gives the output as the hyperplane with the largest margin that separates the data. The homogenous case of this form is given when the bias is zero and the separator goes through the origin. Below is a plots the maximum margin separating hyperplane within a two-class separable dataset using a Support Vector Machine classifier with linear kernel. 

![png](/assets/images/magin_hyperplane.png)


### Hard Margin SVM Code
```python
class HardSVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        return

    def fit(self, X, Y):
        # Check that X and y have correct shape
        X, Y = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.Y_ = Y
        
        self.hardSVM = SVC(kernel='linear', C=1000)
        self.hardSVM.fit(X, Y)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'Y_'])
        # Input validation
        X = check_array(X)
        
        return self.hardSVM.predict(X)
```


## Soft-Margin
Hard-margin uses a rather strong assumption that the training set is linearly separable, this is not always the case. Soft-margin is an extension of hard-margin for data that is not linearly separable. Soft margin SVM uses a slack variables, $$\xi_1, \xi_2,..,\xi_m$$  to penalize a training point when ever it is misclassified.

>  $$ \textbf{input:} (x_1,x_2),...,(x_m,y_m) $$  
$$ \textbf{solve:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ \underset{(w,b,\xi)}{\min} \bigg( \lambda \|w\|^2 + \frac{1}{m} \sum_{i=1}^{m} \xi_i \bigg) $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; s.t. &nbsp; $$  \forall i,  y_i(<w,x_i> + b) \geq 1 - \xi_i $$ &nbsp; and &nbsp; $$ \xi_i \geq 0 $$  
$$ \textbf{output:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ w, b $$

Soft-margin SVM minimizes both the norm of $$w$$ (the margin) and the average of $$\xi_i$$ (corresponding to the violation of the constrains). The regularization term $$ \lambda $$ controls the trade-off between $$\xi_i$$ and $$w$$. The quote below explains bias-variance trade-off of soft-margin SVM.


> Let C be defined as the inverse of the regularization term. The bias-variance trade off of soft-margin SVM are evident at small and large values of C. When C is large, we allow for more violation of margin and although there is less variance (more stability), there is potentially high bias. The opposite holds for small values of C, less violation on training data is allowed, less bias (low training error) but higher variance.

### Soft Margin SVM Code
```python
class SoftSVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        return

    def fit(self, X, Y):
        # Check that X and y have correct shape
        X, Y = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.Y_ = Y
        
        # Stochastic Gradient Descent (SGD) is used to train a soft-margin SVM
        # The loss function is therefore the hinge loss and
        # ... the regularizaiton term will be l2 (Euclidean)
        # The constant multiplied by the regulariztion term is alpha (in the class we used lambda)
        # max_iter is the maximum number of iterations for the optimization
        self.softSVM = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000,tol=0.001, alpha=.1)
        self.softSVM.fit(X, Y)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'Y_'])
        # Input validation
        X = check_array(X)
        
        return self.softSVM.predict(X)
```

## Kernalized SVM 
It is important to note that linearly non-separable features often become linearly separable if they are mapped to a high dimensional feature space. Embedding the input space into some high dimensional feature space makes the learning more expressive but at higher computational complexity and costs. However, using the kernel trick we don't need to compute the feature mapping $$\phi$$ explicitly. I won’t go specifics of kernels in this post but they are used to describe inner products in the feature space. Kernelized SVMs use the kernels of the feature mappings which are easier to compute. Doing so makes it possible to define a complex decision boundary on a high dimensional (even infinite) feature space with efficient computation.

>  $$ \textbf{Learning:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ \underset{\alpha}{\min} \frac{1}{n} \max {0,1 - y^i<\alpha,k(x^i)>} + \lambda\alpha^T K\alpha $$  
$$ \textbf{Predictor for point x:} $$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $$ sgn(<\alpha,k(x)>) $$

### Kernalized SVM Code
Keralized SVM is used with a RBF kernal and tunes its hyperparamteres using GridSearchCV. 

```python
class KernalVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, gamma,demo_param='demo'):
        self.demo_param = demo_param
        self.gamma = gamma
        return

    def svc_param_selection(self,X, y):
        nFoldCV = KFold()
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nFoldCV)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_
    
    def fit(self, X, Y):
        # Check that X and y have correct shape
        X, Y = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(Y)
        self.X_ = X
        self.Y_ = Y

        best_parameters= self.svc_param_selection(X,Y)
        self.kernalVM = SVC(kernel='rbf', gamma = best_parameters['gamma'], 
                            C= best_parameters['C'])
        self.kernalVM.fit(X,Y)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'Y_'])
        # Input validation
        X = check_array(X)
        return self.kernalVM.predict(X)
```
## Digit Recognition with SVM predictors
In this example We use cross-validation to determine the accuracy. In t-fold cross-validation, the data is divided into t subsets. One of the subsets is used for testing, and  the other t-1 subsets (folds) are used for training. The test score is then calculated. We take the average of these t possible ways as a measure for accuracy of the method. For more specifics on cross-validation check out my previous post on Nearest Neighbors. 

```python
# loading the digits data set.
digits = load_digits()

print ("We have", digits.data.shape[0], "data points in total.")
print ("Each data point has", digits.data.shape[1], "features. In fact, each instance is an 8x8 picture of a digit.")
print ("Each digit's target/class/label is one of", list(set(digits.target)))

# We want to turn the problem into a binary classification one.
# Therefore, from the data set, we only pick those digits whose label is either 0 or 1
X = digits.data[(digits.target==0)|(digits.target==1)]
Y = digits.target[(digits.target==0)|(digits.target==1)]
print ("The reamining number of data points is", X.shape[0])
```

    We have 1797 data points in total.
    Each data point has 64 features. In fact, each instance is an 8x8 picture of a digit.
    Each digit's target/class/label is one of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    The reamining number of data points is 360
    


```python
def show_some_digits(X, Y):
    plt.figure(figsize=(20,8))
    for index, (image, label) in enumerate(zip(X[0:30], Y[0:30])):
        plt.subplot(3, 10, index + 1)
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
        plt.title('Target Label: %i\n' % label, fontsize = 10)
    return

def blurr_digit (X, drop_probability):
# This is how we create the noisy data set
    return np.multiply(X, np.random.choice([0, 1], size=(360,64), p=[drop_probability, 1 - drop_probability]))


print ("Hear are some of the original digits (first three rows)...and then their noisy versions")
show_some_digits(X, Y)
plt.show()
X_noisy = blurr_digit (X, 0.6)
show_some_digits(X_noisy, Y)
plt.show()

```

    Hear are some of the original digits (first three rows)...and then their noisy versions
    


![png](/assets/images/output_2_1.png)



![png](/assets/images/output_2_2.png)


```python

# Hard SVM Classifier
hardSVM = HardSVMClassifier()
# Soft SVM classifier
softSVM = SoftSVMClassifier()
#Kernalized VM
kernalVM = KernalVMClassifier(gamma=None)

hardSVM_scores_all = []
softSVM_scores_all = []
kernalVM_scores_all = []

for t in range(2,10):
    
    hardSVM_scores = cross_val_score(hardSVM, X_noisy, Y, cv=t)
    hardSVM_scores_all.append(hardSVM_scores.mean())
#    plot_learning_curve(hardSVM,"hard-svm", X_noisy, Y, cv=t)
    
    softSVM_scores = cross_val_score(softSVM, X_noisy, Y, cv=t)
    softSVM_scores_all.append(softSVM_scores.mean())
#     plot_learning_curve(softSVM,"soft-svm", X_noisy, Y, cv=t)
    
    kernalVM_scores = cross_val_score(kernalVM, X_noisy, Y, cv=t)
    kernalVM_scores_all.append(kernalVM_scores.mean())
#     plot_learning_curve(kernalVM,"kernalVM", X_noisy, Y, cv=t)



plt.figure(figsize=(12,8))
plt.plot(hardSVM_scores_all)
plt.plot(softSVM_scores_all)
plt.plot(kernalVM_scores_all)


plt.legend(['Hard SVM', 'Soft SVM', "Kernalized VM" ])
plt.title('Accuracy of hardSVM vs softSVM vs KernalVM #folds')
plt.xlabel('Number of folds')
plt.ylabel('Accuracy')

plt.show()
```

### A Comparison of SVM Predictors

![png](/assets/images/svm_plot.png)

Kernelized SVM has the best performance of all the classifiers. However, this is also due to the fact that the linear SVM is a special case RBF, and is never more accurate than a properly tuned RBF kernel (as in my implementation). It should be noted that, the computational expense may outweigh the little increase in performance provided by using a non-linear kernel. The implementation uses GridSeachCV to tune the hyper-parameters $$\gamma$$ and C, the penalty parameter. Hyperparameter tuning frequently used a lower value of $\gamma$ which resulted in a lower bias and higher variance. High variance is depicted in the plot of learning curves for KernalVM (try this out yourself,email me if you need help) and is indicated by a gap between the training and CV scores. The training score remains close to its maximum as the number of training examples increase this indicates severe overfitting. Cross-validation score increases over time, which leads to the notion that, in order to reduce the variance, one must reduce the complexity of the model or gather more data. Please note, that 64 features are still miniscule and can be used with non-linear SVM. Consider, n is the number of features and m the number of training examples. In practice, if n is large relative to m 
> $$(n >> m \| n = 10000, m=10..1000)$$ 

logistic regression or linear kernel is used. If n is small, m is intermediate then its best to use SVM with a gaussian kernel such as RBF.

## References
1. Aggarwal, Charu C., et al. On the Surprising Behavior of Distance Metrics in High
Dimensional Space. Database Theory ICDT 2001 Lecture Notes in Computer Science,
2001, pp. 420434., doi:10.1007/3-540-44503-x27
2. Shalev-Shwartz, Shai, and Shai Ben-David. Understanding Machine Learning: from Theory to Algorithms. Cambridge University Press, 2017.
