from collections import defaultdict

import pandas as pd
from systematic_review.converter import write_json_file_with_dict, json_file_to_dict

from feature_selection.algorithms import Filter, AllFeatureSelection, FeatureSelectionAuto
from feature_selection.machine_learning_algorithms import ModelTesting


class FeaturesData:
    def __init__(self, data=None, headers=None, labels=None, type_of_algorithm="all"):
        self.labels = labels
        self.headers = headers
        self.data = data
        self.type_of_algorithm = type_of_algorithm
        self.selected_features_data = defaultdict(list)
        self.feature_selection_algorithms = []

    def features_placeholder(self, method_name, type_of_selection, feature_selection_result):
        self.selected_features_data[method_name].append({"type_of_selection": type_of_selection,
                                                         "feature_selection_result": feature_selection_result})

    def variables_analysis(self, stored_data=None):
        if stored_data:
            data = stored_data
        else:
            data = self.selected_features_data

        counter_dict = dict()
        for method_name, feature_selection_list in data.items():
            for feature_dict in feature_selection_list:
                for i in feature_dict["feature_selection_result"]:
                    if i in counter_dict:
                        counter_dict[i] += 1
                    else:
                        counter_dict[i] = 1
        return counter_dict


class CrossValidationKFold:
    def __init__(self, clf_subset_data, clf_y, n_splits=5):
        self.n_splits = n_splits
        self.clf_subset_data = clf_subset_data
        self.clf_y = clf_y
        self.length = len(self.clf_subset_data)

    def k_fold_steps(self):
        split_index = self.length // self.n_splits

        result = [[0, split_index]]

        temp = []
        for i in range(split_index, self.length, split_index):
            temp.append(i)

        x = temp[0]
        for j in temp[1:]:
            result.append([x, j])
            x = j

        result.append([temp[-1], self.length])
        return result

    def get_all_folds(self):
        kfolds_index_list = self.k_fold_steps()

        kfolds_datasets = []
        for index in kfolds_index_list:
            training = kfolds_index_list.copy()
            training.remove(index)

            clf_y_train = self.clf_subset_data.iloc[index[0]:index[1]]
            clf_y_test = self.clf_y.iloc[index[0]:index[1]]

            if index[0] == 0:
                clf_x_train = self.clf_subset_data.iloc[index[1]:]
                clf_x_test = self.clf_y.iloc[index[1]:]
            elif index == kfolds_index_list[-1]:
                clf_y_train = self.clf_subset_data.iloc[index[0]:index[1]]
                clf_y_test = self.clf_y.iloc[index[0]:index[1]]

                clf_x_train = self.clf_subset_data.iloc[:index[0]]
                clf_x_test = self.clf_y.iloc[:index[0]]
            else:
                df_pre_x = self.clf_subset_data.iloc[:index[0]]
                df_pre_y = self.clf_y.iloc[:index[0]]

                df_post_x = self.clf_subset_data.iloc[index[1]:]
                df_post_y = self.clf_y.iloc[index[1]:]

                clf_x_train = pd.concat([df_pre_x, df_post_x], axis=1)
                clf_x_test = pd.concat([df_pre_y, df_post_y], axis=1)

            kfolds_datasets.append([clf_x_train, clf_x_test, clf_y_train, clf_y_test])

        return kfolds_datasets


def get_subset_data_based_on_columns(data, subset_columns_list):
    return data[subset_columns_list]


class HybridSubsetFeatureSelection:
    def __init__(self, clf_data, clf_y):
        self.clf_y = clf_y
        self.clf_data = clf_data
        self.saved_results = FeaturesData()

    def get_best_subset(self, apply_filter=True):
        if apply_filter:
            modified_data = Filter(self.clf_data).sequential_all()
        else:
            modified_data = self.clf_data

        for number_of_top_features_to_select in range(1, len(modified_data.columns)):
            feature_selection_data = AllFeatureSelection(modified_data,
                                                         self.clf_y).get_names_from_all(
                number_of_top_features_to_select)

            for subset_columns in feature_selection_data:
                if subset_columns in self.saved_results:  # dynamic_programming_dict
                    continue

                subset_data = get_subset_data_based_on_columns(modified_data, subset_columns)

                for dataset in CrossValidationKFold(subset_data, self.clf_y).get_all_folds():

                    ModelTesting(dataset[0], dataset[1], dataset[2], dataset[3]).get_all_models()

        feature_selection_data = FeatureSelectionAuto(modified_data, self.clf_y).get_all()
        for subset_columns in feature_selection_data:
            if subset_columns in self.saved_results:  # dynamic_programming_dict
                continue

            subset_data = get_subset_data_based_on_columns(modified_data, subset_columns)

            for dataset in CrossValidationKFold(subset_data, self.clf_y).get_all_folds():
                ModelTesting(dataset[0], dataset[1], dataset[2], dataset[3]).get_all_models()

    def get_all_feature_selection_info(self):
        pass

    def get_features_ranking(self):
        pass

    def get_test_metrics(self):
        pass

