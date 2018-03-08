
#prediction of new label according to given coordinates and labels

import numpy as np
X=np.array([[-1,-1],[-1,0],[-2,0],[1,1],[2,0],[3,2]]) # an array is taken for coordinates
Y=np.array([1,2,3,4,5,6]) # an another for corresponding labels 
from sklearn.naive_bayes import GaussianNB # importing external modules 
clf = GaussianNB() # creating a classifier
clf.fit(X,Y) # fitting statement we are giving it training data so that it can learn pattern
GaussianNB()
print(clf.predict([[3,1]])) #printing new label for new coordinate

# output-6
# just plot the graph and labels!!!
