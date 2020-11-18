# -*- coding: utf-8 -*-
"""
Data pre-processing module.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from warnings import warn

logger = logging.getLogger(__name__)


class PreProcess:
    """Class to handle the pre-processing of feature data."""

    def __init__(self, features, feature_names=None):
        """
        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame.
        feature_names : str, optional
            Feature names, used if features is an ndarray, by default None
        """

        self._features = features
        self._pd = False
        if isinstance(self._features, pd.DataFrame):
            self._pd = True
            self._feature_names = self._features.columns.tolist()
            if not features.index.is_unique:
                msg = 'DataFrame indices must be unique'
                logger.error(msg)
                raise AttributeError(msg)
        else:
            self._pd = False
            check = (feature_names is not None
                     and len(set(feature_names)) != features.shape[1])
            if check:
                msg = ('The number of feature names ({}) does not match the '
                       'number of features ({})!'
                       .format(len(set(feature_names)), features.shape[1]))
                logger.error(msg)
                raise ValueError(msg)

            self._feature_names = feature_names

    @staticmethod
    def _check_stdev(stdev):
        """
        Check stdev values for 0s or near 0 values, replace with 1s

        Parameters
        ----------
        stdev : int | ndarray
            Normalization stdev value(s)

        Returns
        -------
        stdev : int | ndarray
            Normalization stdev values(s) with 0s replaced with 1s
        """
        zeros = np.isclose(stdev, 0)
        if np.any(zeros):
            msg = ('Standard deviation is ~0 and will be set to 1')
            logger.warning(msg)
            warn(msg)
            if isinstance(zeros, bool):
                stdev = 1
            else:
                stdev[zeros] = 1

        return stdev

    @staticmethod
    def normalize(native_arr, mean=None, stdev=None):
        """
        Normalize features with mean at 0 and stdev of 1.

        Parameters
        ----------
        native_arr : ndarray
            native data
        mean : float | None
            mean to use for normalization
        stdev : float | None
            stdev to use for normalization

        Returns
        -------
        norm_arr : ndarray
            normalized data
        mean : float
            mean used for normalization
        stdev : float
            stdev used for normalization
        """

        if mean is None:
            mean = np.nanmean(native_arr, axis=0)

        if stdev is None:
            stdev = np.nanstd(native_arr, axis=0)
            stdev = PreProcess._check_stdev(stdev)

        norm_arr = native_arr - mean
        norm_arr /= stdev

        return norm_arr, mean, stdev

    @staticmethod
    def unnormalize(norm_arr, mean, stdev):
        """
        Unnormalize data with mean at 0 and stdev of 1.

        Parameters
        ----------
        norm_arr : ndarray
            normalized data
        mean : float
            mean used for normalization
        stdev : float
            stdev used for normalization

        Returns
        -------
        native_arr : ndarray
            native un-normalized data
        """
        native_arr = norm_arr * stdev
        native_arr += mean

        return native_arr

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

    @staticmethod
    def check_one_hot_categories(one_hot_categories, feature_names=None):
        """
        Check one hot features and categories for duplicate names and against
        feature names if provided

        Parameters
        ----------
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories
        feature_names : [type], optional
            Feature names, by default None
        """
        one_hot_features_names = [i for l in one_hot_categories.values()
                                  for i in l]
        names, feature_counts = np.unique(one_hot_features_names,
                                          return_counts=True)
        if any(feature_counts > 1):
            msg = ('one-hot category names have to be unique accross all '
                   'features. The following category names were duplicated:'
                   '\n{}'.format(names[feature_counts > 1]))
            logger.error(msg)
            raise RuntimeError(msg)

        if feature_names is not None:
            one_hot_features = np.array(list(one_hot_categories))
            check = np.isin(one_hot_features, feature_names)
            if not all(check):
                bad_names = one_hot_features[~check]
                msg = ('The following one-hot features do not have valid '
                       'names!\n{}\nMust be one of the available feature '
                       'names:\n{}'.format(bad_names, feature_names))
                logger.error(msg)
                raise RuntimeError(msg)

            final_names = list(set(feature_names) - set(one_hot_categories))
            check = np.isin(one_hot_features_names, final_names)
            if any(check):
                msg = ('The following category names: {} conflict with '
                       'existing feature names'
                       .format(np.array(one_hot_features_names)[check]))
                logger.error(msg)
                raise RuntimeError(msg)

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
            name = self._feature_names[i] if self._feature_names else None

            n = len(self._features)
            if self._pd:
                col = self._features.iloc[:, i].values.reshape((n, 1))
            else:
                col = self._features[:, i].reshape((n, 1))

            if not self._is_one_hot(col, convert_int=convert_int):
                numerical_ind.append(i)
            else:
                logger.debug('One hot encoding {}'.format(name))
                one_hot_ind.append(i)

                if name in categories:
                    cats = [categories[name]]
                    logger.debug('Using categories {} for column {}'
                                 ''.format(cats, name))
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

    def process_one_hot(self, convert_int=False, categories=None,
                        return_ind=False):
        """Process str and int columns in the feature data to one-hot vectors.

        Parameters
        ----------
        convert_int : bool, optional
            Flag to convert integer data to one-hot vectors, by default False
        categories : dict | None, optional
            Categories to use for one hot encoding where a key is the original
            column name in the feature dataframe and value is a list of the
            possible unique values of the feature column. The value list must
            have as many or more entries as unique values in the feature
            column. This will name the feature column headers for the new
            one-hot-encoding if features is a dataframe. Empty dict or None
            results in category names being determined automatically. Format:
                {'col_name1' : ['cat1', 'cat2', 'cat3'],
                 'col_name2' : ['other_cat1', 'other_cat2']}
            by default None
        return_ind : bool, optional
            Return one hot column indices, by default False

        Returns
        -------
        processed : np.ndarray | pd.DataFrame
            Feature data with str and int columns removed and one-hot boolean
            vectors appended as new columns. If features is a dataframe and
            categories is input, the new one-hot columns will be named
            according to categories.
        one_hot_ind : list, optional
            List of numeric column indices in the native data that are
            to-be-transformed into one-hot vectors.
        """

        if categories is None:
            categories = {}
        else:
            self.check_one_hot_categories(categories,
                                          feature_names=self._feature_names)

        one_hot_ind, one_hot_data, numerical_ind = self._get_one_hot_data(
            convert_int=convert_int, categories=categories)

        if not one_hot_ind:
            processed = self._features
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

        if return_ind:
            return processed, one_hot_ind
        else:
            return processed

    @classmethod
    def one_hot(cls, features, feature_names=None, convert_int=False,
                categories=None, return_ind=False):
        """
        Process str and int columns in the feature data to one-hot vectors.

        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame.
        feature_names : str, optional
            Feature names, used if features is an ndarray, by default None
        convert_int : bool, optional
            Flag to convert integer data to one-hot vectors, by default False
        categories : dict | None, optional
            Categories to use for one hot encoding where a key is the original
            column name in the feature dataframe and value is a list of the
            possible unique values of the feature column. The value list must
            have as many or more entries as unique values in the feature
            column. This will name the feature column headers for the new
            one-hot-encoding if features is a dataframe. Empty dict or None
            results in category names being determined automatically. Format:
                {'col_name1' : ['cat1', 'cat2', 'cat3'],
                 'col_name2' : ['other_cat1', 'other_cat2']}
            by default None
        return_ind : bool, optional
            Return one hot column indices, by default False

        Returns
        -------
        processed : np.ndarray | pd.DataFrame
            Feature data with str and int columns removed and one-hot boolean
            vectors appended as new columns. If features is a dataframe and
            categories is input, the new one-hot columns will be named
            according to categories.
        one_hot_ind : list, optional
            List of numeric column indices in the native data that are
            to-be-transformed into one-hot vectors.
        """
        logger.debug('Checking for one-hot items and converting them '
                     'to binary values')
        pp = cls(features, feature_names=feature_names)
        out = pp.process_one_hot(convert_int=convert_int,
                                 categories=categories,
                                 return_ind=return_ind)

        return out
