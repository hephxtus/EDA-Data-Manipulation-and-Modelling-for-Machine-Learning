import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = ['Id', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                   'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                   'TotRmsAbvGrd',
                   'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                   'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
CATAGORICAL_COLUMNS = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
                       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                       'Exterior2nd',
                       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                       'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
                       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'KitchenQual', 'Functional', 'FireplaceQu',
                       'GarageType',
                       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                       'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']


def get_project_root() -> Path:
    # print(Path((os.path.realpath(__file__))).parent.parent)
    return Path((os.path.realpath(__file__))).parent


def get_src_dir() -> str:
    ROOT_DIR = os.path.dirname(get_project_root())
    return os.path.join(ROOT_DIR, "src")


def get_out_dir() -> str:
    ROOT_DIR = os.path.dirname(get_project_root())
    return os.path.join(ROOT_DIR, "out")


def get_data_dir() -> str:
    ROOT_DIR = os.path.dirname(get_project_root())
    # print(os.path.join(ROOT_DIR, "data"))
    return os.path.join(ROOT_DIR, "data")

def read_data():
    """
    Get the data from the data folder
    :return:
    """


    DATA_DIR = get_data_dir()

    print("DATA DIRECTORY:", DATA_DIR)

    df = pd.read_csv(f"{DATA_DIR}\House_Price.csv")

    return df


class classification_helpers:

    @staticmethod
    def process_data(dataset):
        """
        Read the dataset and scale it.
        """
        scaler = StandardScaler()

        X, y, class_labels = [], [], set()

        # Split into features and class labels
        for row in dataset:
            # row is a tuple
            row = list(row)

            # every row is a list of features and class label
            # all but the final column are the list of features
            X.append(row[:-1])

            # class labels are final column
            class_label = row[-1]
            class_labels.add(class_label)

            y.append(row[-1])

        # one-hot encode the class labels
        y = np.array(y)
        y = np.where(y == list(class_labels)[0], 0, 1)

        # Scale the features
        X = scaler.fit_transform(X)

        return X, y

    @staticmethod
    def draw_table(classification_results, headers, title):
        """draw summary table of best means with rows being algorithms and columns being datasets"""

        import pandas as pd
        print(title)
        print('\n')
        vectors = {}
        print(f"saving file to data/{title}.txt...")

        for header in headers:
            vectors[header] = {}
            for algorithm_name, runs in classification_results.items():
                for dataset, hyperparams in runs.items():
                    if dataset == header:
                        best = {'hyperparam': 0, 'mean': 0, 'std': 0}
                        for hyperparam, accuracies in hyperparams.items():
                            if np.mean(accuracies) > best['mean']:
                                best['hyperparam'] = hyperparam
                                best['mean'] = np.mean(accuracies)
                                best['std'] = np.std(accuracies)
                        vectors[header][algorithm_name] = best

        df = pd.DataFrame(
            index=classification_results.keys(),
            columns=headers,
            data=vectors
        )
        print(pathlib.Path('out/'))
        outfile = os.path.join(get_project_root().parent, 'out', title + '.csv')
        df.to_csv(outfile, index=True, encoding='utf-8')

    @staticmethod
    def algorithm_boxplots(classification_results):
        """
        Draw a boxplot of the results.
        """
        print(classification_results)
        for algorithm, vectors in classification_results.items():
            print(f"drawing boxplot for {algorithm}:")
            fig, axs = plt.subplots(nrows=1, ncols=3, )  # sharex=True, sharey=True)
            fig.suptitle(algorithm)

            for x, (dataset_name, hyper_p) in enumerate(vectors.items()):
                axs[x].set_title(dataset_name)

                axs[x].boxplot(hyper_p.values(), labels=hyper_p.keys())

            plt.subplots_adjust(wspace=0.5, hspace=0.5)

            print(f"saving file to data/{algorithm}.png...")
            plt.savefig(pathlib.Path('out/box/' + algorithm + '.png'))
            plt.close()
            print("DONE...\n")

        print("EXITING...\n")


class clustering_helpers:
    @staticmethod
    def plot_clusters(nrows, ncols, loaded, datasets):
        """
        Draw clusters.
        """

        for method, data in loaded.items():
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, )  # sharex=True, sharey=True)
            fig.suptitle(method)
            for x, (dataset, labels) in enumerate(data.items()):
                print(dataset)
                axs[x].set_title(dataset)
                out_data = np.reshape(labels, (labels.shape[0], 1))
                try:
                    data_shape = np.hstack((datasets[dataset][0], out_data))
                except:
                    data_shape = np.hstack((datasets[dataset].data, out_data))
                # axs[x].scatter(data_shape[:, 0], data_shape[:, 1], c=data_shape[:, 2], cmap='viridis')
                axs[x].scatter(x=[d[0] for d in data_shape], y=[d[1] for d in data_shape], c=labels)
            # fig.show()
            fig.savefig('out/scatter/' + method + '.png')
            plt.close()
