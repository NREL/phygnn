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

        if self._pd:
            if not features.index.is_unique:
                raise AttributeError('DataFrame indices must be unique')

    @staticmethod
    def _is_one_hot(arr, convert_int=False):
        """Check if an array of data is to be transformed into a one-hot vector
        by sampling the first datum and checking the type.

        Parameters
        ----------
        arr : np.ndarray
            Array (column) of data to be checked.
        convert_int : bool
            Flag to convert integer data to one-hot vectors.

        Returns
        -------
        one_hot : bool
            True if arr is to be transformed into a one-hot vector.
        """
        if len(arr.shape) == 1:
            sample = arr[0]
        elif len(arr.shape) == 2:
            sample = arr[0, 0]
        else:
            e = 'Cannot process 3D column into one hot'
            logger.error(e)
            raise ValueError(e)

        one_hot = False

        if isinstance(sample, str):
            one_hot = True
        elif np.issubdtype(type(sample), np.integer) and convert_int:
            one_hot = True

        return one_hot

    def _get_one_hot_data(self, convert_int=False, categories=None):
        """Get one hot data and column indexes.

        Parameters
        ----------
        convert_int : bool
            Flag to convert integer data to one-hot vectors.
        categories : dict | None
            Categories to use for one hot encoding where a key is the original
            column name in the feature dataframe and value is a list of the
            possible unique values of the feature column. The value list must
            have as many or more entries as unique values in the feature
            column. This will name the feature column headers for the new
            one-hot-encoding if features is a dataframe. Empty dict or None
            results in category names being determined automatically. Format:
                {'col_name1' : ['cat1', 'cat2', 'cat3'],
                 'col_name2' : ['other_cat1', 'other_cat2']}

        Returns
        -------
        one_hot_ind : list
            List of numeric column indices in the native data that are
            to-be-transformed into one-hot vectors.
        one_hot_data : list
            List of arrays of one hot data columns that are transformations of
            the one_hot_ind columns.
        numerical_ind : list
            List of numeric column indices in the native data that are
            continuous numerical columns that are not to-be-transformed into
            one-hot vectors.
        """

        if categories is None:
            categories = {}

        one_hot_ind = []
        one_hot_data = []
        numerical_ind = []

        for i in range(self._features.shape[1]):
            if self._pd:
                col_name = self._features.columns[i]
            else:
                col_name = None

            n = len(self._features)
            if self._pd:
                col = self._features.iloc[:, i].values.reshape((n, 1))
            else:
                col = self._features[:, i].reshape((n, 1))

            if not self._is_one_hot(col, convert_int=convert_int):
                numerical_ind.append(i)
            else:
                logger.debug('One hot encoding {}'.format(col_name))
                one_hot_ind.append(i)

                if col_name in categories:
                    cats = [categories[col_name]]
                    logger.debug('Using categories {} for column {}'
                                 ''.format(cats, col_name))
                    oh_obj = OneHotEncoder(sparse=False, categories=cats)
                else:
                    oh_obj = OneHotEncoder(sparse=False)

                oh_obj.fit(col)
                one_hot_data.append(oh_obj.transform(col))

        return one_hot_ind, one_hot_data, numerical_ind

    def _make_df_one_hot_cols_labels(self, one_hot_ind, one_hot_data,
                                     categories=None):
        """Make unique column labels for the new one-hot data. This will use
        column labels from categories if available.

        Parameters
        ----------
        one_hot_ind : list
            List of numeric column indices in the native data that are
            to-be-transformed into one-hot vectors.
        one_hot_data : list
            List of arrays of one hot data columns that are transformations of
            the one_hot_ind columns.
        categories : dict | None
            Categories to use for one hot encoding where a key is the original
            column name in the feature dataframe and value is a list of the
            possible unique values of the feature column. The value list must
            have as many or more entries as unique values in the feature
            column. This will name the feature column headers for the new
            one-hot-encoding if features is a dataframe. Empty dict or None
            results in category names being determined automatically. Format:
                {'col_name1' : ['cat1', 'cat2', 'cat3'],
                 'col_name2' : ['other_cat1', 'other_cat2']}

        Returns
        -------
        col_labels : list
            List of string labels corresponding to np.hstack(one_hot_data).
        """

        if categories is None:
            categories = {}

        col_labels = []
        for i, oh_ind in enumerate(one_hot_ind):
            orig_col_label = self._features.columns.values[oh_ind]
            if orig_col_label in categories:
                cat_labels = categories[orig_col_label]

                msg = ('Values in the categories input dict must be a '
                       'list or tuple!')
                assert isinstance(cat_labels, (list, tuple)), msg

                unique_vals = pd.unique(self._features[orig_col_label])
                msg = ('Categories for "{a}" one-hot column had fewer unique '
                       'entries than one-hot encodings! You input these '
                       'categories: {b} but "{a}" has these values: {c}'
                       .format(a=orig_col_label, b=cat_labels, c=unique_vals))
                assert len(cat_labels) >= len(unique_vals), msg

                if isinstance(cat_labels, tuple):
                    cat_labels = list(cat_labels)

                col_labels += cat_labels
            else:
                def_labels = [orig_col_label + '_' + str(k)
                              for k in range(one_hot_data[i].shape[1])]
                col_labels += def_labels

        return col_labels

    def process_one_hot(self, convert_int=False, categories=None):
        """Process str and int columns in the feature data to one-hot vectors.

        Parameters
        ----------
        convert_int : bool
            Flag to convert integer data to one-hot vectors.
        categories : dict | None
            Categories to use for one hot encoding where a key is the original
            column name in the feature dataframe and value is a list of the
            possible unique values of the feature column. The value list must
            have as many or more entries as unique values in the feature
            column. This will name the feature column headers for the new
            one-hot-encoding if features is a dataframe. Empty dict or None
            results in category names being determined automatically. Format:
                {'col_name1' : ['cat1', 'cat2', 'cat3'],
                 'col_name2' : ['other_cat1', 'other_cat2']}

        Returns
        -------
        processed : np.ndarray | pd.DataFrame
            Feature data with str and int columns removed and one-hot boolean
            vectors appended as new columns. If features is a dataframe and
            categories is input, the new one-hot columns will be named
            according to categories.
        """

        if categories is None:
            categories = {}

        one_hot_ind, one_hot_data, numerical_ind = self._get_one_hot_data(
            convert_int=convert_int, categories=categories)

        if not one_hot_ind:
            return self._features

        else:
            if self._pd:
                num_df = self._features.iloc[:, numerical_ind]
                col_labels = self._make_df_one_hot_cols_labels(one_hot_ind,
                                                               one_hot_data,
                                                               categories)
                one_hot_df = pd.DataFrame(np.hstack(one_hot_data),
                                          columns=col_labels,
                                          index=self._features.index)
                processed = num_df.join(one_hot_df)
                assert processed.shape[0] == num_df.shape[0] == \
                    one_hot_df.shape[0]

            else:
                processed = np.hstack((self._features[:, numerical_ind],
                                       np.hstack(one_hot_data)))
                assert processed.shape[0] == self._features.shape[0]

            processed = processed.astype(np.float32)
            return processed
