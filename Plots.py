import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import seaborn as sns


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
