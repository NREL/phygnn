"""
Data pre-processing module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class PreProcess:
    """Class to handle the pre-processing of feature data."""

    def __init__(self, features):
        """
        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame.
        """
        self._features = features
        self._pd = False
        if isinstance(self._features, pd.DataFrame):
            self._pd = True

    def process_one_hot(self, convert_int=False):
        """Process str and int columns in the feature data to one-hot vectors.

        Parameters
        ----------
        convert_int : bool
            Flag to convert integer data to one-hot vectors.

        Returns
        -------
        processed : np.ndarray | pd.DataFrame
            Feature data with str and int columns removed and one-hot boolean
            vectors appended as new columns.
        """

        one_hot_ind = []
        one_hot_data = []
        numerical_ind = []

        for i in range(self._features.shape[1]):

            n = len(self._features)
            if self._pd:
                col = self._features.iloc[:, i].values.reshape((n, 1))
            else:
                col = self._features[:, i].reshape((n, 1))

            sample = col[0, 0]
            one_hot = False

            if isinstance(sample, str):
                one_hot = True
            elif np.issubdtype(type(sample), np.integer) and convert_int:
                one_hot = True

            if one_hot:
                one_hot_ind.append(i)
                oh_obj = OneHotEncoder(sparse=False)
                oh_obj.fit(col)
                one_hot_data.append(oh_obj.transform(col))
            else:
                numerical_ind.append(i)

        if one_hot_ind:
            if self._pd:
                processed = self._features.iloc[:, numerical_ind]
                cols = [[self._features.columns[j] + '_' + str(k)
                         for k in range(one_hot_data[i].shape[1])]
                        for i, j in enumerate(one_hot_ind)]
                cols = [a for sublist in cols for a in sublist]
                one_hot_df = pd.DataFrame(np.hstack(one_hot_data),
                                          columns=cols,
                                          index=self._features.index)
                processed = processed.join(one_hot_df)
            else:
                processed = np.hstack((self._features[:, numerical_ind],
                                       np.hstack(one_hot_data)))

        processed = processed.astype(np.float32)
        return processed
