Q1. Mathematical Formula for a Linear SVM
For a linear SVM, the decision function is represented as:

𝑓
(
𝑥
)
=
sign
(
𝑤
⋅
𝑥
+
𝑏
)
f(x)=sign(w⋅x+b)

where:

𝑤
w is the weight vector,
𝑥
x is the input vector (features of the data point),
𝑏
b is the bias term,
sign
sign is the sign function which determines the class label.
Q2. Objective Function of a Linear SVM
The objective function of a linear SVM aims to maximize the margin between the support vectors. It can be expressed as:

min
⁡
𝑤
,
𝑏
1
2
∥
𝑤
∥
2
min 
w,b
​
  
2
1
​
 ∥w∥ 
2
 

subject to the constraints:

𝑦
𝑖
(
𝑤
⋅
𝑥
𝑖
+
𝑏
)
≥
1
∀
𝑖
y 
i
​
 (w⋅x 
i
​
 +b)≥1∀i

where 
𝑥
𝑖
x 
i
​
  are the training samples, 
𝑦
𝑖
y 
i
​
  are their labels (+1 or -1), 
𝑤
w is the weight vector, and 
𝑏
b is the bias term.

Q3. Kernel Trick in SVM
The kernel trick in SVM allows it to handle nonlinear decision boundaries by implicitly mapping the input vectors into a higher-dimensional feature space. It avoids the computationally expensive task of explicitly calculating the coordinates of the data in that space. The kernel function 
𝐾
(
𝑥
𝑖
,
𝑥
𝑗
)
K(x 
i
​
 ,x 
j
​
 ) computes the dot product of the mapped feature vectors in the higher-dimensional space without explicitly constructing them.

Q4. Role of Support Vectors in SVM
Support vectors are the data points that lie closest to the decision boundary (margin). They play a crucial role in defining the decision boundary because the SVM classifier is determined by these support vectors. These vectors support the optimal hyperplane and influence the classifier's performance.

Example:
Consider a binary classification problem where we have two classes, +1 and -1, and a set of data points. The support vectors are the data points that are nearest to the decision boundary (hyperplane) that separates the two classes. These points are crucial because they define the maximum margin hyperplane, which maximizes the separation between the classes.

Q5. Illustration of Hyperplane, Marginal Plane, Soft Margin, and Hard Margin in SVM
Hyperplane:

In SVM, the hyperplane is the decision boundary that separates the data points of different classes.
Example: In a 2D space, the hyperplane is a line; in 3D, it's a plane.
Marginal Plane:

The marginal plane in SVM is the region parallel to the hyperplane that defines the margins.
Example: For a linear SVM, the marginal planes are the planes that are equidistant from the hyperplane and define the margin.
Soft Margin and Hard Margin:

Hard Margin: Requires all training data points to be correctly classified with no margin violations.
Soft Margin: Allows some margin violations (misclassifications) to achieve a wider margin and better generalization.
Example: In a soft margin SVM, the decision boundary (hyperplane) may tolerate a few misclassifications (violations) to achieve better separation and generalization.
Q6. SVM Implementation through Iris Dataset
Here's how we can implement SVM using the Iris dataset:

Steps:
Load the Iris dataset and Split into Training and Testing Sets:
python
Copy code
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Train a Linear SVM Classifier:
python
Copy code
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a linear SVM classifier
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)

# Train the classifier
svm_clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = svm_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM on the testing set: {accuracy:.2f}')
Plot the Decision Boundaries:
python
Copy code
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

# Plot decision boundary for two features
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=svm_clf, legend=2)

# Adding axes annotations
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Decision Boundary of Linear SVM on Iris Dataset')
plt.show()
Tuning the Regularization Parameter C:
python
Copy code
# Try different values of C
C_values = [0.1, 1.0, 10.0]
for C in C_values:
    svm_clf = SVC(kernel='linear', C=C, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy with C={C}: {accuracy:.2f}')
This completes the implementation and comparison task for a linear SVM classifier using the Iris dataset. The provided code loads the dataset, splits it into training and testing sets, trains the SVM classifier, evaluates its performance, plots decision boundaries, and tunes the regularization parameter 
𝐶
C.
