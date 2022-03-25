from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from typing import Union

class ColumnsConcatenation(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, columns: Union[str, list] = None, level: int = None, max_cardinality: int = None):
        # save the categorical columns and level of combinations internally in the class
        self.columns = columns
        self.level = level
        self.max_cardinality = max_cardinality

    # infer caterogical columns
    def _get_categorical_columns(self, X):
        return X.select_dtypes(include=['category', 'object']).columns.to_list()

    # fit function
    def fit(self, X, y=None):
        # infer categorical columns, if 'auto' is selected
        if self.columns == 'auto':
            self.columns = self._get_categorical_columns(X)
        # if a list is passed as columns, use it as it is
        elif isinstance(self.columns, list):
            pass
        # if nothing is passed use all columns (e.g. a ColumnTransformer with make_column_selector is used before)
        else:
            self.columns = X.columns.to_list()

        # remove the columns with cardinality higher than max_cardinality
        if self.max_cardinality:
            cols_to_remove =(X[self.columns].nunique() > self.max_cardinality).to_list()
            self.columns = [cols for (cols, remove) in zip(self.columns, cols_to_remove) if not remove]
        return self

    # transform function
    def transform(self, X):

        cat_combinations = list()

        if self.level == None:
            for i in range(2,len(self.columns)+1):
                cat_combinations.extend(list(combinations(self.columns, i)))

            for comb in cat_combinations:
                X['_'.join(comb)] = X[comb[0]].str.cat(X[list(comb[1:])].astype(str), sep="_").astype("category")
        else:
            for i in range(2,self.level+1):
                cat_combinations.extend(list(combinations(self.columns, i)))

            for comb in cat_combinations:
                X['_'.join(comb)] = X[comb[0]].str.cat(X[list(comb[1:])].astype(str), sep="_").astype("category")

        return X


