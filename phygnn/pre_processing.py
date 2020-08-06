"""
Data pre-processing module.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class PreProcess:
    """Class to handle the pre-processing of feature data."""

    def __init__(self, features, categories={}):
        """
        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame.
        categories : dict
            Categories to use for one hot encoding. Format:
                {
                    'col_name1' : ['cat1', 'cat2', 'cat3'],
                    'col_name2' : ['other_cat1', 'other_cat2']
                }
        """
        if not features.index.is_unique:
            raise AttributeError('DataFrame indices must be unique')

        self._categories = categories
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

        for i, col_name in enumerate(self._features.columns):

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
                logger.debug('One hot encoding {}'.format(col_name))
                one_hot_ind.append(i)
                if col_name in self._categories:
                    categories = [self._categories[col_name]]
                    logger.debug('Using categories {} for column {}'
                                 ''.format(categories, col_name))
                    oh_obj = OneHotEncoder(sparse=False, categories=categories)
                else:
                    oh_obj = OneHotEncoder(sparse=False)
                oh_obj.fit(col)
                one_hot_data.append(oh_obj.transform(col))
            else:
                numerical_ind.append(i)

        if not one_hot_ind:
            return self._features

        if one_hot_ind:
            if self._pd:
                num_df = self._features.iloc[:, numerical_ind]
                cols = [[self._features.columns[j] + '_' + str(k)
                         for k in range(one_hot_data[i].shape[1])]
                        for i, j in enumerate(one_hot_ind)]
                cols = [a for sublist in cols for a in sublist]
                one_hot_df = pd.DataFrame(np.hstack(one_hot_data),
                                          columns=cols,
                                          index=self._features.index)
                processed = num_df.join(one_hot_df)

                assert processed.shape[0] == num_df.shape[0] == \
                    one_hot_df.shape[0]
            else:
                processed = np.hstack((self._features[:, numerical_ind],
                                       np.hstack(one_hot_data)))
                assert processed.shape[0] == self.features.shape[0]

            processed = processed.astype(np.float32)
            return processed
