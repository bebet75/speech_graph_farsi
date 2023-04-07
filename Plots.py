import numpy as np
from numpy import where
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import plot_confusion_matrix, classification_report, log_loss, roc_curve, roc_auc_score, ConfusionMatrixDisplay


def simple_plot(X, Y):
    # plotting if there is only two features
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = where(Y == class_value)
        # create scatter of these samples
        print(X.values[row_ix, 0])
        plt.scatter(X.values[row_ix, 0], X.values[row_ix, 1])
    plt.show()


def sen_word_rel(features):
    # word count
    wc = features[features['diagnosis'] ==
                  'Schizo']['word_count'].values.tolist()
    sc = features[features['diagnosis'] ==
                  'Schizo']['sentence_count'].values.tolist()
    wcs = np.sum(wc)
    scs = np.sum(sc)
    wq1 = [x / y for (x, y) in zip(wc, sc)]
    ww1 = wcs / scs
    # print(sorted(wq1))
    wc = features[features['diagnosis'] ==
                  'Control']['word_count'].values.tolist()
    sc = features[features['diagnosis'] ==
                  'Control']['sentence_count'].values.tolist()
    wcs = np.sum(wc)
    scs = np.sum(sc)
    wq2 = [x / y for (x, y) in zip(wc, sc)]
    ww2 = wcs / scs
    # print(sorted(wq2))
    print('Schizo', ww1)
    print('Control', ww2)


def Bar_plot(features_df, feature_name):
    # bar chart plot
    # Import data
    features_df1 = features_df[features_df['diagnosis']
                               == "Schizo"][[feature_name]]
    features_df2 = features_df[features_df['diagnosis']
                               == "Control"][[feature_name]]
    features_df3 = pd.concat(
        [features_df1, features_df2], axis=1, ignore_index=True, sort=False)
    features_df3 = features_df3.apply(lambda x: pd.Series(x.dropna().values))
    features_df3.rename(columns={0: 'Schizo'}, inplace=True)
    features_df3.rename(columns={1: 'Control'}, inplace=True)
    for n in range(1, features_df3.columns.shape[0] + 1):
        features_df3.rename(
            columns={
                f"data{n}": f"Experiment {n}"},
            inplace=True)
    features_df3.head()
    vals, names, xs = [], [], []
    for i, col in enumerate(features_df3.columns):
        vals.append(features_df3[col].dropna().values)
        names.append(col)
        # adds jitter to the data points - can be adjusted
        xs.append(
            np.random.normal(
                i / 2 + 0.5,
                0.04,
                features_df3[col].dropna().values.shape[0]))
    plt.figure(figsize=(5, 15))
    palette = ['r', 'g']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    ### '''## Set style options here #####
    sns.set_style(
        "darkgrid", {
            "grid.color": ".6", "grid.linestyle": ":"})  # "white","dark","darkgrid","ticks"
    boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
    flierprops = dict(marker='o', markersize=1, linestyle='none')
    whiskerprops = dict(color='#00145A')
    capprops = dict(color='#00145A')
    medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')

    palette = ['#FF2709', '#09FF10']
    plt.boxplot(
        vals,
        positions=[
            0.5,
            1],
        widths=0.3,
        labels=names,
        notch=False,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
        medianprops=medianprops,
        showmeans=False)
    plt.xlabel('Diagnosis', fontweight='normal', fontsize=14)
    plt.ylabel(feature_name, fontweight='normal', fontsize=14)
    sns.despine(bottom=True)  # removes right and top axis lines
    # plt.axhline(y=65, color='#ff3300', linestyle='--', linewidth=1,
    # label='Threshold Value')
    plt.legend(
        bbox_to_anchor=(
            0.31,
            1.06),
        loc=2,
        borderaxespad=0.,
        framealpha=1,
        facecolor='white',
        frameon=True)
    plt.show()


def Dis_plot(features_df, feature_name):
    # Distributution plot
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    # Import data
    x1 = features_df[features_df['diagnosis'] == "Schizo"][feature_name]
    x2 = features_df[features_df['diagnosis'] == "Control"][feature_name]
    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(x1, color="dodgerblue", label="Schizo", **kwargs)
    sns.distplot(x2, color="orange", label="Control", **kwargs)
    # plt.xlim(50,75)
    plt.axvline(x=x1.mean(), color='blue', ls='--')
    plt.axvline(x=x2.mean(), color='orange', ls='--')
    plt.xlabel(feature_name)
    plt.legend()


def Stats(features_df, feature_name):
    # Statistic numbers
    x1 = features_df[features_df['diagnosis'] == "Schizo"][feature_name]
    x2 = features_df[features_df['diagnosis'] == "Control"][feature_name]
    print('mean_Schz', np.mean(x1))
    print('STD_Schz', np.std(x1))
    print(np.mean('mean_Cntrl', x2))
    print(np.std('STD_Cntrl', x2))
    print('ttest_ind', stats.ttest_ind(a=x1, b=x2, equal_var=True))
    print('kruskal', stats.kruskal(x1, x2))
    print('ranksums', stats.ranksums(x1, x2))


def reg_plot(X,Y):
    ###### Step 4: Standardization
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Initiate scaler
    sc = StandardScaler()
    # Standardize the training dataset
    X_train_transformed = pd.DataFrame(sc.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
    # Standardized the testing dataset
    X_test_transformed = pd.DataFrame(sc.transform(X_test),index=X_test.index, columns=X_test.columns)
    # Summary statistics after standardization
    X_train_transformed.describe().T
    # Summary statistics before standardization 
    X_train.describe().T
    ###### Step 5: Logistic Regression With No Regularization
    # Check default values
    LogisticRegression()
    # Run model
    logistic = LogisticRegression(penalty='none', random_state=0).fit(X_train_transformed, y_train)
    # Make prediction
    logistic_prediction = logistic.predict(X_test_transformed)
    # Get predicted probability
    logistic_pred_Prob = logistic.predict_proba(X_test_transformed)[:,1]
    # Get the false positive rate and true positive rate
    fpr,tpr, _= roc_curve(y_test, logistic_pred_Prob)
    # Get auc value
    auc = roc_auc_score(y_test, logistic_pred_Prob)
    # Plot the chart
    plt.plot(fpr,tpr,label="area="+str(auc))
    plt.legend(loc=4)
    plt.show()
    # Caclulate log loss
    log_loss(y_test,logistic_pred_Prob)
    # Confusion matrix
    plot_confusion_matrix(logistic, X_test_transformed, y_test)
    # Performance report
    print(classification_report(y_test, logistic_prediction, digits=3))
    # Model coefficients
    LogisticCoeff = pd.concat([pd.DataFrame(X_test_transformed.columns),pd.DataFrame(np.transpose(logistic.coef_))], axis = 1)
    LogisticCoeff.columns=['Variable','Coefficient']
    LogisticCoeff['Coefficient_Abs']=LogisticCoeff['Coefficient'].apply(abs)
    print(LogisticCoeff.sort_values(by='Coefficient_Abs', ascending=False))
    ###### Step 6: LASSO
    # Run model
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=0).fit(X_train_transformed, y_train)
    # Make prediction
    lasso_prediction = lasso.predict(X_test_transformed)
    # Get predicted probability
    lasso_pred_Prob = lasso.predict_proba(X_test_transformed)[:,1]
    # Get the false positive rate and true positive rate
    fpr,tpr, _= roc_curve(y_test,lasso_pred_Prob)
    # Get auc value
    auc = roc_auc_score(y_test,lasso_pred_Prob)
    # Plot the chart
    plt.plot(fpr,tpr,label="area="+str(auc))
    plt.legend(loc=4)
    plt.show()
    # Calculate log loss
    log_loss(y_test,lasso_pred_Prob)
    # Confusion matrix
    plot_confusion_matrix(lasso, X_test_transformed, y_test)
    # Performance report
    print(classification_report(y_test, lasso_prediction, digits=3))
    # Model coefficients
    lassoCoeff = pd.concat([pd.DataFrame(X_test_transformed.columns),pd.DataFrame(np.transpose(lasso.coef_))], axis = 1)
    lassoCoeff.columns=['Variable','Coefficient']
    lassoCoeff['Coefficient_Abs']=lassoCoeff['Coefficient'].apply(abs)
    print(lassoCoeff.sort_values(by='Coefficient_Abs', ascending=False))
    ###### Step 7: Ridge
    # Run model
    ridge = LogisticRegression(penalty='l2', random_state=0).fit(X_train_transformed, y_train)
    # Make prediction
    ridge_prediction = ridge.predict(X_test_transformed)
    # Get predicted probability
    ridge_pred_Prob = ridge.predict_proba(X_test_transformed)[:,1]
    # Get the false positive rate and true positive rate
    fpr,tpr, _= roc_curve(y_test,ridge_pred_Prob)
    # Get auc value
    auc = roc_auc_score(y_test,ridge_pred_Prob)
    # Plot the chart
    plt.plot(fpr,tpr,label="area="+str(auc))
    plt.legend(loc=4)
    plt.show()
    # Calculate log loss
    log_loss(y_test,ridge_pred_Prob)
    # Confusion matrix
    plot_confusion_matrix(ridge, X_test_transformed, y_test)
    # Performance matrix
    print(classification_report(y_test, ridge_prediction, digits=3))
    # Model coefficients
    ridgeCoeff = pd.concat([pd.DataFrame(X_test_transformed.columns),pd.DataFrame(np.transpose(ridge.coef_))], axis = 1)
    ridgeCoeff.columns=['Variable','Coefficient']
    ridgeCoeff['Coefficient_Abs']=ridgeCoeff['Coefficient'].apply(abs)
    print(ridgeCoeff.sort_values(by='Coefficient_Abs', ascending=False))
    ###### Step 8: Elastic Net
    # Run model
    elasticNet = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=0).fit(X_train_transformed, y_train)
    # Make prediction
    elasticNet_prediction = elasticNet.predict(X_test_transformed)
    # Get predicted probability
    elasticNet_pred_Prob = elasticNet.predict_proba(X_test_transformed)[:,1]
    # Get the false positive rate and true positive rate
    fpr,tpr, _ = roc_curve(y_test,elasticNet_pred_Prob)
    # Get auc value
    auc = roc_auc_score(y_test,elasticNet_pred_Prob)
    # Plot the chart
    plt.plot(fpr,tpr,label="area="+str(auc))
    plt.legend(loc=4)
    plt.show()
    # Calculate log loss
    log_loss(y_test,elasticNet_pred_Prob)
    # Confusion matrix
    plot_confusion_matrix(elasticNet, X_test_transformed, y_test)
    # Performance report
    print(classification_report(y_test, elasticNet_prediction, digits=3))
    # Model coefficients
    elasticNetCoeff = pd.concat([pd.DataFrame(X_test_transformed.columns),pd.DataFrame(np.transpose(elasticNet.coef_))], axis = 1)
    elasticNetCoeff.columns=['Variable','Coefficient']
    elasticNetCoeff['Coefficient_Abs']=elasticNetCoeff['Coefficient'].apply(abs)
    elasticNetCoeff.sort_values(by='Coefficient_Abs', ascending=False)


def plot_classification_results(clf, X, y, list_of_features):
    title = ''
    feature1 = list_of_features[0]
    feature2 = list_of_features[1]
    # Divide dataset into training and testing parts
    X = X[list_of_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #y=y['diagnosis']=="Schizo"
    # Fit the data with classifier.
    clf.fit(X_train, y_train)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    h = .01  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[feature1].min() - 0.1, X[feature1].max() + 0.1
    y_min, y_max = X[feature2].min() - 0.1, X[feature2].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    fig, ax = plt.subplots()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    matplotlib.pyplot.xlabel(feature1)
    matplotlib.pyplot.ylabel(feature2)
    # Plot also the training points
    plt.scatter(X_train[feature1], X_train[feature2], c=y_train, cmap=cmap_bold, s=80)

    y_predicted = clf.predict(X_test)
    score1 = clf.score(X, y)
    score2 = clf.score(X_test, y_test)
    scatter = ax.scatter(X_test[feature1], X_test[feature2], c=y_predicted, alpha=0.5, cmap=cmap_bold,marker='^', s =80)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #ax.legend(['g','r'],['Control','Schizo'],loc="lower right", title="Classes")
    title = 'Full Data Accuracy : ' + str(round(score1, 2)) + '  /  Test Data Accuracy : ' + str(round(score2, 2))
    plt.title(title)
