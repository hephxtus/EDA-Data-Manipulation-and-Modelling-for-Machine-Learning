import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from common.utils import NUMERIC_COLUMNS, CATAGORICAL_COLUMNS, read_data, get_data_dir, get_out_dir


def preprocess(df):
    """
    Determine and describe the data preprocessing steps applied to the provided dataset, e.g. handle
    missing data, encoding categorical data, normalise the data if necessary, and/or remove any unnecessary instances, these could be redundant instances, outliers or non-effective instances and so forth. Show the process
    in your code/workflow. Submit a copy of the processed dataset (in CSV format).
    :param df:
    :return:
    """
    _NUMERIC_COLUMNS = NUMERIC_COLUMNS[:]
    _CATAGORICAL_COLUMNS = CATAGORICAL_COLUMNS[:]
    new_df = df.copy()
    # Remove unnecessary columns
    new_df.drop(['Id'], axis=1, inplace=True)
    _NUMERIC_COLUMNS.remove('Id')
    _NUMERIC_COLUMNS.remove('SalePrice')

    # remove features with missing values > 0.8
    to_drop = new_df.columns[new_df.isnull().mean() > 0.8]
    new_df.drop(to_drop, axis=1, inplace=True)
    [_CATAGORICAL_COLUMNS.remove(col) if col in _CATAGORICAL_COLUMNS else _NUMERIC_COLUMNS.remove(col) for col in to_drop]
    # new_df.dropna(thresh=0.8, axis=1, inplace=True)
    # new_df.dropna(axis=0, thresh=len(new_df) * 0.8, inplace=True)

    # fill missing values with median of collumn
    new_df.fillna(0, inplace=True)

    # Normalise the data
    new_df[_NUMERIC_COLUMNS] = preprocessing.scale(new_df[_NUMERIC_COLUMNS])

    # # Encode categorical data
    new_df[_CATAGORICAL_COLUMNS] = new_df[_CATAGORICAL_COLUMNS].astype('category')
    new_df[_CATAGORICAL_COLUMNS] = new_df[_CATAGORICAL_COLUMNS].apply(lambda x: x.cat.codes)

    # Remove redundant instances
    new_df.drop_duplicates(inplace=True)

    # standardise the data
    columns = new_df.columns
    X = pd.DataFrame(StandardScaler().fit_transform(new_df[columns[:-1]]), columns=columns[:-1])

    # remove rows with outliers
    # print(new_df[(np.abs(stats.zscore(new_df)) < 5).all(axis=1)].index)

    return X, new_df[columns[-1]]


from sklearn.decomposition import PCA


def Dimensionality_Reduction(X, y):
    """
    Utilise two different dimensionality reduction techniques to identify which features are irrelevant
    and/or redundant to predicting the house price. Report the dimension reduction process and remove redundant/irrelevant data. Show the process in your code/workflow.

    :param df:
    :return:
    """

    def _perform_reduction(X, y, method, n_components=10):
        """
        Perform dimensionality reduction on the provided dataset using the specified method.
        :param X:
        :param y:
        :param method:
        :param n_components:
        :return:
        """
        # initialise model
        model = method(n_components=n_components, )

        # fit model
        model.fit(X, y)

        # transform data
        x_3d = model.transform(X)

        return x_3d

    # PCA
    model = PCA(n_components=10)
    model.fit_transform(X)
    _variance = model.explained_variance_

    # create variance graph
    plt.figure(figsize=(8, 6))
    plt.bar(range(10), _variance, alpha=0.5, align='center', label='individual variance')
    plt.legend()
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    # Isomap
    X_iso = _perform_reduction(X, y, Isomap, )
    plt.title('Isomap')
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y)
    plt.ylim([-40, 40])
    plt.show()
    # Locally Linear Embedding
    X_lle = _perform_reduction(X, y, LocallyLinearEmbedding)
    plt.title('Locally Linear Embedding')
    plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y)
    plt.show()

    return X_iso, X_lle


def evaluate_reduction(features_new, y, title):
    """
    Evaluate the performance of the dimensionality reduction technique on the provided dataset.
    :param features_new:
    :param y:
    :return:
    """
    # Split the data into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(features_new, y, test_size=0.3, random_state=309)
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=309)

    # initialise regression models
    linreg_clf = LinearRegression()
    ridge_clf = Ridge(alpha=0.5)
    rfr_clf = RandomForestRegressor(n_estimators=100, random_state=309)

    # train the models
    linreg_clf.fit(X_train, y_train)
    ridge_clf.fit(X_train, y_train)
    rfr_clf.fit(X_train, y_train)

    linreg_pred = linreg_clf.predict(X_val)
    ridge_pred = ridge_clf.predict(X_val)
    rfr_pred = rfr_clf.predict(X_val)

    # calculate the mean squared error
    linreg_mse = mean_squared_error(y_val, linreg_pred)
    ridge_mse = mean_squared_error(y_val, ridge_pred)
    rfr_mse = mean_squared_error(y_val, rfr_pred)

    # calculate the r2 score
    linreg_r2 = r2_score(y_val, linreg_pred)
    ridge_r2 = r2_score(y_val, ridge_pred)
    rfr_r2 = r2_score(y_val, rfr_pred)

    print('Linear Regression MSE: {}'.format(linreg_mse))
    print('Linear Regression R2: {}%'.format(linreg_r2))

    print('Ridge Regression MSE: {}'.format(ridge_mse))
    print('Ridge Regression R2: {}%'.format(ridge_r2))

    print('Random Forest Regression MSE: {}'.format(rfr_mse))
    print('Random Forest Regression R2: {}%'.format(rfr_r2))

    plt.scatter(linreg_pred, y_val, c='r', label='Linear Regression')
    plt.scatter(ridge_pred, y_val, c='g', label='Ridge Regression')
    plt.scatter(rfr_pred, y_val, c='b', label='Random Forest Regression')

    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Actual Price ($)')
    plt.legend()
    plt.title('Regression Performance on {}'.format(title))
    plt.ylim([0, y_val.max()])
    plt.xlim([0, y_val.max()])
    plt.show()

    # calculate the cross-validated scores
    scores_linear = cross_val_score(linreg_clf, X_val, y_val, scoring='neg_mean_squared_error', verbose=-1, cv=cv,
                                    n_jobs=-1)
    scores_ridge = cross_val_score(ridge_clf, X_val, y_val, scoring='neg_mean_squared_error', verbose=-1, cv=cv,
                                   n_jobs=-1)
    scores_rfr = cross_val_score(rfr_clf, X_val, y_val, scoring='neg_mean_squared_error', verbose=-1, cv=cv, n_jobs=-1)

    return [scores_linear.mean(), scores_ridge.mean(), scores_rfr.mean()]
    # print("Accuracy {} Regression: {}".format("Linear", scores_linear.mean()))
    # print("Accuracy {} Regression: {}".format("Ridge", scores_ridge.mean()))
    # print("Accuracy {} Regression: {}".format("Random Forest", scores_rfr.mean()))

def main():
    data_dir = get_data_dir()

    df = read_data()
    # preprocess data
    X, y = preprocess(df)
    df = pd.merge(X, y, left_index=True, right_index=True)

    # output data to csv
    df.to_csv(f'{data_dir}/preprocessed.csv', index=False)

    # Dimensionality Reduction
    iso, lle = Dimensionality_Reduction(X, y)
    results = pd.DataFrame(index=['Linear Regression', 'Ridge Regression', 'Random Forest Regression'],)

    print('Original feature #:', X.shape[1])
    print('Reduced feature #:', iso.shape[1], lle.shape[1])
    print("Initial Accuracy (mean squared error):")
    results['Original'] = evaluate_reduction(X, y, 'Original')

    print("Isometric Accuracy (mean squared error):")
    results['Isometric'] = evaluate_reduction(iso, y, 'Isometric')
    # evaluate_reduction(iso, y, "Isometric")

    print("Locally Linear Embedding Accuracy (mean squared error):")
    results['Locally Linear Embedding'] = evaluate_reduction(lle, y, 'Locally Linear Embedding')
    # evaluate_reduction(lle, y, "LLE")

    print(results)
    out_dir = get_out_dir()
    results.to_csv(f'{out_dir}/Regression_Results.csv')


if __name__ == "__main__":
    main()
