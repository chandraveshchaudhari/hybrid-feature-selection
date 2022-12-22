from collections import defaultdict


def sample_func(word: str):
    """
    this is a sample func for demonstration of numpy docstring and import methods

    Parameters
    ----------
    word : str
        this is any string word to pass in function

    Returns
    -------
    str
        this is output with some added text to word

    """
    output = f"this is how this function print: {word}"
    return output


class HybridFeatureSelection:
    def __init__(self, data=None, headers=None, labels=None, type_of_algorithm="all"):
        self.labels = labels
        self.headers = headers
        self.data = data
        self.type_of_algorithm = type_of_algorithm
        self.selected_features_data = defaultdict(list)

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
