# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report, log_loss, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

# A loop over some of the possible Classifiers
classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(128,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(64,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(256,)),
    MLPClassifier(alpha=0.2, max_iter=1000, hidden_layer_sizes=(32, 64)),
    LogisticRegression(solver='newton-cg', random_state=0),
    LogisticRegression(penalty='l1', solver='liblinear', random_state=0, max_iter=200),
    LogisticRegression(random_state=0),
    LogisticRegression(penalty='l2', random_state=0)
]

'''
classifiers = [

    DecisionTreeClassifier(max_depth=8),
    AdaBoostClassifier(),
    LogisticRegression(solver='newton-cg', random_state=0, max_iter = 1000),
    LogisticRegression( solver='liblinear', max_iter = 1000),
    LogisticRegression(max_iter = 1000),
    LogisticRegression( solver='liblinear' ,random_state=0, max_iter = 1000),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=1)
]
'''


def classifier_full(X, Y, classifiers):
    # a dict to save output scores
    mean_scores = {}
    max_scores = {}
    sc = StandardScaler()

    # iterate over classifiers
    for clf in classifiers:
        # A loop over some of the possible Classifiers
        all_scores1 = []
        all_scores2 = []

        for i in range(50):
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2)
            X_train_transformed = pd.DataFrame(
                sc.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns)
            # Standardized the testing dataset
            X_test_transformed = pd.DataFrame(
                sc.transform(X_test),
                index=X_test.index,
                columns=X_test.columns)
            X_transformed = pd.DataFrame(
                sc.transform(X), index=X.index, columns=X.columns)
            # fitting train data
            clf.fit(X_train_transformed, y_train)
            # Scoring the Output
            all_scores1.append(clf.score(X_test_transformed, y_test))
            all_scores2.append(clf.score(X_transformed, Y))

        # appending the score to scores dictionary
        mean_scores[clf] = np.mean(all_scores1)
        max_scores[clf] = max(all_scores2)

    # print all of scores
    print('mean', mean_scores)
    print('max', max_scores)


def classifier_one(X, Y, clf, ):
    sc = StandardScaler()
    result_max = 0
    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2)
        X_train_transformed = pd.DataFrame(
            sc.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns)
        # Standardized the testing dataset
        X_transformed = pd.DataFrame(
            sc.transform(X),
            index=X.index,
            columns=X.columns)
        X_test_transformed = pd.DataFrame(
            sc.transform(X_test),
            index=X_test.index,
            columns=X_test.columns)
        # fitting train data
        clf.fit(X_train_transformed, y_train)
        # Scoring the Output
        if result_max < clf.score(X_transformed, Y):
            result_max = clf.score(X_transformed, Y)
            result_mean = clf.score(X_test_transformed, y_test)
            best_clf = clf
            best_sc = sc
    '''
    filename = 'best_clf.sav'
    pickle.dump(best_clf, open(filename, 'wb'))
    filename = 'best_sc.sav'
    pickle.dump(best_sc, open(filename, 'wb'))
    '''
    print('max accuracy is ', result_max)
    print('mean accuracy is ', result_mean)
    return best_clf, result_max, result_mean, best_sc


def classifier_new_data(best_clf, x, sc):
    #loaded_model = pickle.load(open(filename, 'rb'))
    #loaded_model = pickle.load(open(filename, 'rb'))
    #sc = StandardScaler()
    x = pd.DataFrame(sc.transform(x), index=x.index, columns=x.columns)
    acc = best_clf.predict(x)
    if acc == 1:
        result = 'Schizo'
    elif acc == 0:
        result = 'Control'

    return result
