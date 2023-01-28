from collections import defaultdict

import pandas as pd
from systematic_review.converter import records_list_to_dataframe

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
    def __init__(self, clf_subset_data, clf_y, n_splits=4):
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
        # problem in fold 2>>>> showing null values
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

                clf_x_train = pd.concat([df_pre_x, df_post_x])
                clf_x_test = pd.concat([df_pre_y, df_post_y])

            kfolds_datasets.append([clf_x_train, clf_x_test, clf_y_train, clf_y_test])

        return kfolds_datasets


def get_subset_data_based_on_columns(data, subset_columns_list):
    return data[subset_columns_list]


def add_data_in_tree_dict(entry_point_dict_node,
                          feature_selection_data=None, modified_data=None, clf_y=None):
    for feature_algorithm_name, subset_columns in feature_selection_data.items():
        if subset_columns is dict:
            subset_columns = list(subset_columns.keys())

        if subset_columns is None:
            continue

        if tuple(subset_columns) in entry_point_dict_node:  # dynamic_programming_dict
            continue

        entry_point_dict_node[tuple(subset_columns)] = dict()
        entry_point_dict_node[tuple(subset_columns)][feature_algorithm_name] = dict()

        subset_data = get_subset_data_based_on_columns(modified_data, subset_columns)

        models_testing_with_cross_validation(subset_data, clf_y,
                                             entry_point_dict_node[tuple(subset_columns)][feature_algorithm_name])


def models_testing_with_cross_validation(clf_data=None, clf_y=None, output_data=None):
    result = output_data if output_data else {}

    cv_number = 1
    for dataset in CrossValidationKFold(clf_data, clf_y).get_all_folds():
        cv_name = f"fold_{cv_number}"
        print(cv_name)
        cv_number += 1

        metric_data = ModelTesting(dataset[0], dataset[1], dataset[2], dataset[3]).get_all_models()

        result[cv_name] = metric_data

    return result


def get_records_from_models_testing_with_cross_validation(clf_data=None, clf_y=None, output_data=None):
    result = models_testing_with_cross_validation(clf_data, clf_y, output_data)

    record_list = []
    for cv_name in result.keys():
        for ml in result[cv_name].keys():
            record = {'Cross validation': cv_name,
                      'Machine Learning Algorithm': ml,
                      **result[cv_name][ml]
                      }
            record_list.append(record)
    return record_list


class HybridSubsetFeatureSelection:
    def __init__(self, clf_data=None, clf_y=None, path='Hybrid_subset_feature_selection_data.xlsx'):
        self.path = path
        self.clf_y = clf_y
        self.clf_data = clf_data
        self.saved_results = dict()

    def generate_subsets(self, apply_filter=True, number_of_top_features_to_select_start=1,
                         number_of_top_features_to_select_end=None):
        if apply_filter:
            modified_data = Filter(self.clf_data).sequential_all()
        else:
            modified_data = self.clf_data

        modified_columns = modified_data.columns
        if not number_of_top_features_to_select_end:
            number_of_top_features_to_select_end = len(modified_columns)
        self.saved_results[tuple(modified_columns)] = dict()

        for number_of_top_features_to_select in range(number_of_top_features_to_select_start,
                                                      number_of_top_features_to_select_end):
            print(number_of_top_features_to_select)
            feature_selection_data = AllFeatureSelection(modified_data,
                                                         self.clf_y,
                                                         number_of_top_features_to_select).get_names_from_all()
            add_data_in_tree_dict(self.saved_results[tuple(modified_columns)], feature_selection_data,
                                  modified_data, self.clf_y)

        feature_selection_data = FeatureSelectionAuto(modified_data, self.clf_y).get_all()

        add_data_in_tree_dict(self.saved_results[tuple(modified_columns)], feature_selection_data,
                              modified_data, self.clf_y)

        self.save_info()

    def create_records_list(self):
        records_list = []
        # [tuple(subset_columns)][feature_algorithm_name][cv_name]
        for modified_cols in self.saved_results.keys():

            for subsets in self.saved_results[modified_cols].keys():

                for fs in self.saved_results[modified_cols][subsets].keys():

                    for cv in self.saved_results[modified_cols][subsets][fs].keys():

                        for mls in self.saved_results[modified_cols][subsets][fs][cv].keys():
                            record = {'After Filter Columns': modified_cols,
                                      'Subset': subsets,
                                      'Subset Length': len(subsets),
                                      'Feature Selection Algorithm': fs,
                                      'Cross validation': cv,
                                      'Machine Learning Algorithm': mls,
                                      **self.saved_results[modified_cols][subsets][fs][cv][mls]
                                      }

                            records_list.append(record)

        return records_list

    def save_info(self, path=None):
        if path:
            return save_records_list_to_excel(self.create_records_list(), path)
        return save_records_list_to_excel(self.create_records_list(), self.path)

    def get_features_ranking(self):
        pass

    def get_test_metrics(self):
        pass


def save_records_list_to_excel(data, path="generated_excel_file.xlsx"):

    return records_list_to_dataframe(data).to_excel(path)
