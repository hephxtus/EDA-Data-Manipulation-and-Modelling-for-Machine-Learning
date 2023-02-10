import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

from src.Part2 import preprocess, Dimensionality_Reduction
from common.utils import NUMERIC_COLUMNS, CATAGORICAL_COLUMNS, read_data, get_data_dir

from common.utils import get_out_dir


def extended_regression(X, y, title):
    """
    Use two other regression techniques to build prediction models for the house price with the same preprocessed data
    using in Part 2. Discuss the prediction results, compared with the regression techniques used in Part 2 and identify
    which technique is more suitable for this question and provide your justifications.

    :param Xiso:
    :param Xlle:
    :return:
    """

    # split the data into train and test
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=309)
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)

    def Gauassian_Process_Regression():
        """
        Use a Gaussian Process Regression model to predict the house price.
        :param X:
        :param y:
        :return:
        """
        kernel = RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=309)

        return gpr

    # initialize regression models
    logreg_clf = LogisticRegression()
    gaureg_clf = Gauassian_Process_Regression()

    # train the models
    logreg_clf.fit(X_train, y_train)
    gaureg_clf.fit(X_train, y_train)

    # gather predictions
    pred_log = logreg_clf.predict(X_val)
    pred_gau = gaureg_clf.predict(X_val)

    # graph the predictions
    plt.scatter(pred_gau, y_val, color='red', label="Gaussian Process Regression")
    plt.scatter(pred_log, y_val, color='blue', label='Logistic Regression')
    plt.legend()
    plt.title('Extended Regression Performance on {}'.format(title))
    plt.ylim([0, y_val.max()])
    plt.xlim([0, y_val.max()])
    plt.show()

    # evaluate the models
    scores_log = cross_val_score(logreg_clf, X_val, y_val, scoring='neg_mean_squared_error', verbose=1, cv=cv,
                                 n_jobs=-1)
    scores_gaureg = cross_val_score(gaureg_clf, X_val, y_val, scoring='neg_mean_squared_error', verbose=1, cv=cv,
                                    n_jobs=-1)
    print("Accuracy {} Regression: {}".format("Logistic Regression", scores_log.mean()))
    print("Accuracy {} Regression: {}".format("Gaussian Process Regression", scores_gaureg.mean()))
    return scores_log.mean(), scores_gaureg.mean()


def main():
    # get preprocessed data
    DATA_DIR = get_data_dir()
    DATA = pd.read_csv(f"{DATA_DIR}\preprocessed.csv")

    # split the data into body (X) and target (y)
    X = DATA.drop(['SalePrice'], axis=1)
    y = DATA['SalePrice']

    # get the reduced data for training
    Xiso, Xlle = Dimensionality_Reduction(X, y)

    # run the regression
    results = pd.DataFrame(index=['Logistic Regression', 'Gaussian Process Regression'])
    results['Original'] = extended_regression(X, y, 'Original')
    results['Isomap'] = extended_regression(Xiso, y, 'Isomap')
    results['LLE'] = extended_regression(Xlle, y, 'LLE')

    print(results)

    out_dir = get_out_dir()
    results.to_csv(f'{out_dir}/Regression_Results.csv', mode='a', header=False)


if __name__ == '__main__':
    main()
