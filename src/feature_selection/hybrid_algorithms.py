import re
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

        if tuple(subset_columns) not in entry_point_dict_node:
            entry_point_dict_node[tuple(subset_columns)] = dict()
        if feature_algorithm_name not in entry_point_dict_node[tuple(subset_columns)]:
            entry_point_dict_node[tuple(subset_columns)][feature_algorithm_name] = dict()

        subset_data = get_subset_data_based_on_columns(modified_data, subset_columns)

        models_testing_with_cross_validation(subset_data, clf_y,
                                             entry_point_dict_node[tuple(subset_columns)][feature_algorithm_name])


def models_testing_with_cross_validation(clf_data, clf_y, output_data):
    cv_number = 1
    for dataset in CrossValidationKFold(clf_data, clf_y).get_all_folds():
        cv_name = f"fold_{cv_number}"
        print(cv_name)
        cv_number += 1

        metric_data = ModelTesting(dataset[0], dataset[1], dataset[2], dataset[3]).get_all_models()

        output_data[cv_name] = metric_data

    return output_data


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

        if tuple(modified_columns) not in self.saved_results:
            self.saved_results[tuple(modified_columns)] = dict()

        for number_of_top_features_to_select in range(number_of_top_features_to_select_start,
                                                      number_of_top_features_to_select_end):
            string_writer(f"Iteration : {number_of_top_features_to_select} \n")
            feature_selection_data = AllFeatureSelection(modified_data,
                                                         self.clf_y,
                                                         number_of_top_features_to_select).get_names_from_all()
            add_data_in_tree_dict(self.saved_results[tuple(modified_columns)], feature_selection_data,
                                  modified_data, self.clf_y)
            self.save_info(f"iteration_{number_of_top_features_to_select}.xlsx")

        feature_selection_data = FeatureSelectionAuto(modified_data, self.clf_y).get_all()
        string_writer(f"auto feature selection \n")

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


class GetResultsFromHybridSubsetFeatureSelection:
    def __init__(self, path='Hybrid_subset_feature_selection_data.xlsx'):
        self.data = pd.read_excel(path)
        self.df_dict = self.data.to_dict('records')

    def get_all_subset(self):

        tree = dict()

        for record in self.df_dict:
            #     print(record)

            metric = {'Accuracy Score': record['Accuracy Score'],
                      'Precision Score': record['Precision Score'],
                      'Recall Score': record['Recall Score']
                      }

            if not record['Subset'] in tree:
                tree[record['Subset']] = dict()

            if not record['Cross validation'] in tree[record['Subset']]:
                tree[record['Subset']][record['Cross validation']] = dict()

            tree[record['Subset']][record['Cross validation']][record['Machine Learning Algorithm']] = metric

        compress_dict = dict()
        for subset in tree:
            if subset not in compress_dict:
                compress_dict[subset] = dict()
            for cr in tree[subset]:
                length = len(tree[subset])
                #         print(length)
                for ml in tree[subset][cr]:
                    if ml not in compress_dict[subset]:

                        compress_dict[subset][ml] = tree[subset][cr][ml]
                        for key, value in compress_dict[subset][ml].items():
                            compress_dict[subset][ml][key] = value / length

                    #                 print(compress_dict)
                    else:
                        for key, value in tree[subset][cr][ml].items():
                            #                     print(key, value)
                            compress_dict[subset][ml][key] += value / length
        #                 print(compress_dict)

        subset_with_best_metric = []

        for subset in compress_dict:
            records_data = []

            for ml in compress_dict[subset]:
                record_dict = dict()
                record_dict['Machine Learning Algorithm'] = ml
                record_dict.update(compress_dict[subset][ml])
                records_data.append(record_dict)

            record_df = pd.DataFrame.from_dict(records_data)

            sorted_df = record_df.sort_values(by=['Accuracy Score', 'Precision Score', 'Recall Score'], ascending=False)

            rec = sorted_df[:1].to_dict('records')
            rec[0].update({'Subset': subset})
            #     print(rec[0])
            subset_with_best_metric.append(rec[0])

        last_df = pd.DataFrame.from_dict(subset_with_best_metric)
        neworder = [last_df.columns[-1], *last_df.columns[:-1]]
        rearrange_df = last_df.reindex(columns=neworder)
        rearrange_df = rearrange_df.sort_values(by=['Accuracy Score', 'Precision Score', 'Recall Score'],
                                                ascending=False)
        rearrange_df.to_excel('best_subset.xlsx')
        # return the first row of the dataframe
        return rearrange_df

    def get_best_subset(self):
        variables = []
        subset = self.get_all_subset().iloc[0]['Subset']
        financial_variables = re.findall(r"(?<=')[^']+(?=')", subset)
        for i in financial_variables:
            if i != ', ':
                variables.append(i)

        return variables

    def get_best_subset_dataframe(self):
        variables_list = self.get_best_subset()
        records = []
        for variable in variables_list:
            records.append({'Financial Variable': variable})
        return pd.DataFrame.from_dict(records)

    def get_features_importance_based_on_subset_selection(self):
        main_dict = dict()

        for record_dict in self.df_dict:
            if not record_dict['Subset Length'] in main_dict:
                main_dict[record_dict['Subset Length']] = dict()

            financial_variables = re.findall(r"(?<=')[^']+(?=')", record_dict['Subset'])
            for i in financial_variables:
                if i != ', ':
                    if i in main_dict[record_dict['Subset Length']]:
                        main_dict[record_dict['Subset Length']][i] += 1
                    else:
                        main_dict[record_dict['Subset Length']][i] = 1

        unique = set()
        importance_dict_based_on_subset = dict()

        for subset_length in sorted(main_dict):
            for financial_var in main_dict[subset_length]:
                if financial_var in unique:
                    pass
                else:
                    unique.add(financial_var)
                    if subset_length in importance_dict_based_on_subset:
                        importance_dict_based_on_subset[subset_length].append(financial_var)
                    else:
                        importance_dict_based_on_subset[subset_length] = [financial_var]
        return importance_dict_based_on_subset, main_dict

    def get_features_importance_based_on_subset_selection_dataframe(self):
        records = []
        for subset_length in self.get_features_importance_based_on_subset_selection()[0]:
            strip_list_financial_variables = [i.strip() for i in self.get_features_importance_based_on_subset_selection()[0][
                                subset_length]]
            financial_variables = ", ".join(list(strip_list_financial_variables))
            records.append({'Subset Length': subset_length,
                            'Financial Variable': financial_variables})

        return pd.DataFrame.from_dict(records)

    def get_features_importance_based_on_count_of_selection(self):
        count_dict = dict()
        main_dict = self.get_features_importance_based_on_subset_selection()[1]

        for record_dict in self.df_dict:
            if not record_dict['Subset Length'] in count_dict:
                main_dict[record_dict['Subset Length']] = dict()

            financial_variables = re.findall(r"(?<=')[^']+(?=')", record_dict['Subset'])
            for i in financial_variables:
                if i != ', ':
                    if i in count_dict:
                        count_dict[i] += 1
                    else:
                        count_dict[i] = 1

        return sorted(count_dict.items(), key=lambda item: item[1], reverse=True)

    def get_features_importance_based_on_count_of_selection_dataframe(self):
        importance_dict_based_on_subset = self.get_features_importance_based_on_count_of_selection()
        records = []
        for financial_var in importance_dict_based_on_subset:

            records.append({'Financial Variable': financial_var[0], 'count of selection in subset': financial_var[1]})
        return pd.DataFrame.from_dict(records)


def save_records_list_to_excel(data, path="generated_excel_file.xlsx"):
    return records_list_to_dataframe(data).to_excel(path)


def string_writer(data_string, path='log_file.txt'):
    with open(path, 'a+') as file:
        file.writelines(data_string)
