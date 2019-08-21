from sklearn import tree
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# CHALLENGE - create 3 more classifiers...
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# loop through all the classifiers and run the exercise for each; print results for each
for clf in classifiers:
   

    # [height, weight, shoe_size]
    X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
        [190, 90, 47], [175, 64, 39],
        [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

    Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
        'female', 'male', 'male']


    # CHALLENGE - ...and train them on our data
    clf = clf.fit(X, Y)

    prediction = clf.predict([[190, 70, 43]])

    # CHALLENGE compare their reusults and print the best one!

    print(str(clf) + " Prediction: " + str(prediction))
    
