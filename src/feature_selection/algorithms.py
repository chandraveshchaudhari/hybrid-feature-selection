"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from feature_selection import sample_func
"""
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from skrebate import ReliefF, MultiSURF, TuRF
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURFstar
from sklearn.feature_selection import VarianceThreshold


class Filter:
    def __init__(self, data):
        self.data = data

    def variance_filter(self, dataset=None, threshold=0.0, inplace=False):
        sel = VarianceThreshold(threshold)
        if dataset:
            if inplace:
                sel.fit_transform(dataset)
                return dataset
            modified_data = dataset.copy()
            return sel.fit_transform(modified_data)
        else:
            if inplace:
                sel.fit_transform(self.data)
                return self.data
            modified_data = self.data.copy()
            return sel.fit_transform(modified_data)

    def correlation_filter(self, dataset=None, threshold=0.99):
        """Hall, M. A. (2000). Correlation-based feature selection of discrete and numeric class machine learning

        Parameters
        ----------
        dataset
        threshold

        Returns
        -------

        """

        if dataset:
            # create correlation  matrix
            corr_matrix = dataset.corr().abs()

            # select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find index of columns with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

            # drop the columns
            modified_data = dataset.drop(to_drop, axis=1)
            return modified_data
        else:
            # create correlation  matrix
            corr_matrix = self.data.corr().abs()

            # select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find index of columns with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

            # drop the columns
            modified_data = self.data.drop(to_drop, axis=1)
            return modified_data

    def sequential_all(self, variance_threshold=0.0, correlation_threshold=0.99):
        modified_data = self.variance_filter(self.data, variance_threshold)
        return self.correlation_filter(modified_data, correlation_threshold)


def converting_feature_importance_to_sorted_dict(headers, fs_feature_importances):
    result = dict()

    for feature_name, feature_score in zip(headers, fs_feature_importances):
        result[feature_name] = feature_score
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


class FeatureSelectionNamesOut:
    def __init__(self, clf_data, clf_y, number_of_top_features_to_select=None):
        self.clf_y = clf_y
        self.clf_data = clf_data
        self.k = number_of_top_features_to_select
        self.labels = clf_y.values

    def get_select_k_best(self):
        sel = SelectKBest(k=self.k).fit(self.clf_data, self.clf_y)
        return list(sel.get_feature_names_out(self.clf_data.columns))

    def recursive_feature_elimination_svc(self):
        svc = SVC(kernel="linear", C=1)
        sel = RFE(estimator=svc, n_features_to_select=self.k, step=1)
        sel.fit(self.clf_data, self.clf_y)
        return list(sel.get_feature_names_out(self.clf_data.columns))

    def recursive_feature_elimination_relief_f(self):
        fs = RFE(ReliefF(), n_features_to_select=self.k, step=0.5)
        fs.fit(self.clf_data, self.labels)
        return list(fs.get_feature_names_out())

    def get_sequential_feature_selector_k_neighbors_classifier(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        sel = SequentialFeatureSelector(knn, n_features_to_select=self.k)
        sel.fit(self.clf_data, self.clf_y)
        return list(sel.get_feature_names_out(self.clf_data.columns))

    def get_sequential_feature_selector_logistic_regression_classifier(self):
        lclf = LogisticRegression()
        sel = SequentialFeatureSelector(lclf, k_features=self.k, forward=True, verbose=1,
                                        scoring='neg_mean_squared_error')
        sel.fit(self.clf_data, self.clf_y)
        return list(sel.k_feature_names_)


class FeatureSelectionNamesScore:
    def __init__(self, clf_data, clf_y, number_of_top_features_to_select=None):
        self.clf_y = clf_y
        self.clf_data = clf_data
        self.k = number_of_top_features_to_select
        self.features = clf_data.values
        self.labels = clf_y.values
        self.headers = clf_data.columns

    # names and scores
    def get_relief_f(self):
        fs = ReliefF(n_features_to_select=self.k, n_neighbors=100)
        fs.fit(self.features, self.labels)

        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)

    def get_surf(self):
        fs = SURF(n_features_to_select=self.k)
        fs.fit(self.features, self.labels)

        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)

    def get_surf_star(self):
        fs = SURFstar(n_features_to_select=self.k)
        fs.fit(self.features, self.labels)
        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)

    def get_multi_surf(self):
        fs = MultiSURF(n_features_to_select=self.k)
        fs.fit(self.features, self.labels)
        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)

    def get_multi_surf_star(self):
        fs = MultiSURFstar(n_features_to_select=self.k)
        fs.fit(self.features, self.labels)
        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)

    def get_turf(self):
        fs = TuRF(core_algorithm="ReliefF", n_features_to_select=self.k, pct=0.5, verbose=True)
        fs.fit(self.features, self.labels, self.headers)
        return converting_feature_importance_to_sorted_dict(self.headers, fs.feature_importances_)


class FeatureSelectionAuto:
    def __init__(self, clf_data, clf_y):
        self.clf_y = clf_y
        self.clf_data = clf_data

    # no need of k
    def select_from_model_l1_based(self):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.clf_data, self.clf_y)
        sel = SelectFromModel(lsvc, prefit=True)
        return list(sel.get_feature_names_out(self.clf_data.columns))

    def select_from_model_tree_based(self):
        clf = ExtraTreesClassifier(n_estimators=50).fit(self.clf_data, self.clf_y)

        sel = SelectFromModel(clf, prefit=True)
        return list(sel.get_feature_names_out(self.clf_data.columns))

    def get_all(self):
        return [self.select_from_model_l1_based(), self.select_from_model_tree_based()]


class AllFeatureSelection:
    def __init__(self, clf_data, clf_y, number_of_top_features_to_select=None):
        self.clf_y = clf_y
        self.clf_data = clf_data
        self.k = number_of_top_features_to_select
        self.labels = clf_y.values

    def get_names_out(self, number_of_top_features_to_select):
        sel = FeatureSelectionNamesOut(self.clf_data, self.clf_y, number_of_top_features_to_select)
        sel_result_list = [sel.get_sequential_feature_selector_logistic_regression_classifier(),
                           sel.get_select_k_best(),
                           sel.get_sequential_feature_selector_k_neighbors_classifier(),
                           sel.recursive_feature_elimination_relief_f(),
                           sel.recursive_feature_elimination_svc()]
        return sel_result_list

    def get_names_score_out(self, number_of_top_features_to_select):
        sel = FeatureSelectionNamesScore(self.clf_data, self.clf_y, number_of_top_features_to_select)
        sel_result_list = [sel.get_multi_surf(),
                           sel.get_multi_surf_star(),
                           sel.get_relief_f(),
                           sel.get_turf(),
                           sel.get_surf(),
                           sel.get_surf_star()]
        return sel_result_list

    def get_names_from_all(self, number_of_top_features_to_select):
        result = self.get_names_out(number_of_top_features_to_select).append(
                 self.get_names_score_out(number_of_top_features_to_select))
        return result

