# data manipulation
# data viz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from common.utils import NUMERIC_COLUMNS, CATAGORICAL_COLUMNS, read_data


def Data_Understanding(df):
    """
    Perform an initial EDA on the given data to gain an understanding of the data. The analyses
    should explore the data from four different aspects including:
    :return:
    """

    def summary_stats(df):
        """
        Describe the summary statistics about the data including number of instances, number of features, hod many
        categorical and numerical features, respectively.
        """
        print("")
        print("SUMMARY STATISTICS")
        print("NUMBER OF INSTANCES: ", len(df.index))
        print("NUMBER OF FEATURES: ", len(df.columns))
        print("NUMBER OF CATEGORICAL FEATURES: ", len(CATAGORICAL_COLUMNS))
        print("NUMBER OF NUMERICAL FEATURES: ", len(NUMERIC_COLUMNS))
        print("")

        print("DETAILS:\n", df.info())

    def correlation_matrix(columns):
        """
        Find the top 5 numerical features highly correlated with the target variable (“SalePrice”) according to the
        pearson correlation, report the correlation values.
        """
        print("")
        print("CORRELATION MATRIX")
        print(columns)
        corr = df[columns].corr('pearson')[df.columns[-1]][:-1].sort_values(ascending=False)[:5]
        print(corr)
        print("")
        return corr

    def feature_distribution(features):
        """
        Plot the distributions of these 5 numerical features found in the previous question and the target variable
        using histograms with 10 bins, one for each feature/variable, describe the shape of their distributions with
        skewness and kurtosis (use Scipy for obtaining skewness and kurtosis values if you cannot do it in Orange),
        and tell two patterns from the histograms accordingly.
        """
        print("")
        print("FEATURE DISTRIBUTION")
        for feature in features:
            plt.hist(df[feature], bins=10)
            plt.title(feature)
            plt.show()
            print(feature, "skewness: ", np.round(np.abs(df[feature].skew()), 2))
            print(feature, "kurtosis: ", np.round(np.abs(df[feature].kurtosis()), 2))
            print("")
            plt.close()

    def missing_data():
        """
        Check for missing values. Is there any missing values in the data? write a paragraph to briefly summarise
        the missing information regarding how many features contain missing values and at what percent.
        """
        print("")
        print("MISSING DATA")
        null_vals = df.isnull()
        print("")
        print("NUMBER OF FEATURES WITH MISSING VALUES:", len(null_vals.columns[null_vals.any()]))
        print("PERCENTAGE OF FEATURES WITH MISSING DATA: ",
              round(len(null_vals.columns[null_vals.any()]) / len(null_vals.columns) * 100, 2), "%")
        print("TOTAL AMOUNT OF MISSING DATA: ", null_vals.sum().sum())
        print("PERCENTAGE OF TOTAL DATA MISSING: ",
              round(null_vals.sum().sum() / (len(df.columns) * len(df.index)) * 100, 2), "%")
        print("")
        # compute percentage of missing values
        missing = null_vals.sum().sort_values(ascending=False)
        percent_missing = round(missing / len(df), 3)
        print("The following features are missing >50% of their values\n",
              percent_missing[missing > 0.5 * len(df)])

    print("a) Summary Statistics")
    summary_stats(df)

    print("b) top 5 numerical features")
    top_5 = correlation_matrix(NUMERIC_COLUMNS)

    print("c) feature distribution")
    feature_distribution(top_5.index.values)

    print("d) missing data")
    missing_data()

    return top_5.index.values


def Business_Understanding(df: pd.DataFrame):
    """
    Investigate the business understanding questions based on your exploration of the data. Two key business
    understanding questions (or business objectives) are “what factors affect the house price?” and “how do these
    factors affect the house price?”/“in which way do the factors affect the house price?”
    :return:
    """

    def Random_Forest_Regression(X, y):
        """
        Perform a logistic regression on the data to identify the key factors that affect the house price.
        """
        print("")
        print("LOGISTIC REGRESSION")

        rf_reg = RandomForestRegressor()
        rf_reg.fit(X, y)

        print("FEATURE IMPORTANCE")
        importance = pd.DataFrame({"feature": X.columns, "importance": rf_reg.feature_importances_}).sort_values("importance",
                                                                                                     ascending=False)
        print(importance)
        # plot the feature importance
        plt.bar([x for x in range(len(importance.importance))], importance.importance)
        plt.xticks(range(len(importance.feature)), importance.feature, rotation=90)
        plt.show()

    def Permutation_Classification(X, y):
        """
        Perform a permutation classification to identify the key factors that affect the house price.
        """
        print("")
        print("PERMUTATION CLASSIFICATION")

        model = KNeighborsClassifier()
        model.fit(X, y)

        results = permutation_importance(model, X, y, scoring='accuracy', n_repeats=10, random_state=0)
        importance = pd.DataFrame({"feature": X.columns, "importance": results.importances_mean}).sort_values(
            "importance",
            ascending=False)
        print(importance)
        # plot the feature importance
        plt.bar([x for x in range(len(importance.importance))], importance.importance)
        plt.xticks(range(len(importance.feature)), importance.feature, rotation=90)
        plt.show()

        # # plot the feature importance
        # plt.bar([x for x in range(len(importance))], importance)
        # plt.xticks(range(len(importance)), X.columns, rotation=45)
        # plt.show()

    # get numeric columns from dataframe
    numeric_columns = df[NUMERIC_COLUMNS].copy()

    # fill missing values with the median of the column
    numeric_columns = numeric_columns.fillna(0)

    # strip the target variable and id from the numeric columns & set the target variable to be the SalePrice column
    X = numeric_columns.drop(['SalePrice', 'Id'], axis=1)
    y = numeric_columns["SalePrice"]

    print("")
    print("Logistic Regression for feature importance")
    Random_Forest_Regression(X, y)
    print("")
    print("Permutation Classification for feature importance")
    Permutation_Classification(X, y)
    print("a) What factors affect the house price?")
    print("")
    print("b) How do these factors affect the house price?")
    print("")
    print("c) In which way do the factors affect the house price?")
    print("")


def EDA_CLUSTERING(df: pd.DataFrame, top_5: list):
    """
    (15 marks) EDA using clustering is very useful for understanding the important characteristics of the data.
    Provide a further EDA on the dataset using Hierarchical clustering on the 5 numerical features found in 1(b)
    to answer the question — “Does the house prices vary by neighborhood?”. Report the output dendrogram and
    any other plots and show how do they help you to answer the question.

    :param df:
    :return:
    """

    from scipy.cluster.hierarchy import dendrogram, linkage
    def hierarchical_clustering(top_5):
        """
        Perform a hierarchical clustering on the data to identify the key factors that affect the house price.
        """
        print("")
        print("HIERARCHICAL CLUSTERING")
        headers = ['SalePrice', 'Neighborhood']
        for feat in top_5:
            headers.append(feat)
        X = df[headers]

        # convert Neighborhood to numerical variable
        neighborhood_names = X['Neighborhood'].tolist()
        lab_encoder = preprocessing.LabelEncoder()
        # X_ = X.copy()
        X['Neighborhood'] = lab_encoder.fit_transform(X['Neighborhood'])

        # perform hierarchical clustering on the data
        hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        labels = hierarchical_cluster.fit_predict(X[X.columns.difference(['SalePrice'])])

        # draw scatter plot of the data
        sns.scatterplot(x='Neighborhood',
                        y='SalePrice',
                        data=X,
                        hue=labels).set_title('SalePrice by Neighborhood Cluster')

        plt.show()

        # pairplot of all features against Neighborhood
        sns.pairplot(X, hue="Neighborhood", vars=top_5)
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.show()

        # Draw the dendrogram
        linkage_matrix = linkage(X, 'ward', metric='euclidean')
        plt.figure(figsize=(15, 5))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30, show_contracted=True)
        plt.show()

    hierarchical_clustering(top_5)
    pass


def main():
    # apply some cool styling
    plt.style.use("ggplot")
    rcParams['figure.figsize'] = (12, 6)

    # load the data
    df = read_data()

    # initial exploration of the data
    top_5 = Data_Understanding(df)
    Business_Understanding(df)

    # EDA using clustering
    EDA_CLUSTERING(df, top_5)


if __name__ == '__main__':
    main()
