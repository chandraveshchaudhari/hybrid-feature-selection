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
    def __init__(self, data, headers, labels, type_of_algorithm="all"):
        self.labels = labels
        self.headers = headers
        self.data = data
        self.type_of_algorithm = type_of_algorithm
        self.selected_features_data = dict()

    def features_placeholder(self, method_name, type_of_selection, feature_selection_result):
        self.selected_features_data[method_name] = {"type_of_selection": type_of_selection,
                                                    "feature_selection_result": feature_selection_result}

        return self.selected_features_data


if __name__ == '__main__':
    pass
