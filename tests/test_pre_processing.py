"""
Test the pre processing module.
"""
import numpy as np
import pandas as pd

from phygnn.utilities.pre_processing import PreProcess

index = pd.date_range('20180101', '20190101', freq='5min')
A = pd.DataFrame({'f1': ['a', 'b', 'c', 'd', 'a', 'c', 'a'],
                  'f2': np.arange(7) * 0.333,
                  'f3': np.arange(7, dtype=int)}, index=index[:7])


def test_one_hot_encoding():
    """Process str and int one hot columns and verify outputs."""
    proc = PreProcess(A)
    out = proc.process_one_hot(convert_int=False)

    assert (out.iloc[:, -4:].sum(axis=1) == np.ones(7)).all()
    assert out['f1_0'].values[0] == 1.0
    assert out['f1_0'].values[1] == 0.0
    assert out['f1_0'].values[-3] == 1.0
    assert out['f1_0'].values[-1] == 1.0
    assert out['f1_1'].values[1] == 1.0
    assert out['f1_1'].values[0] == 0.0
    assert out['f1_2'].values[2] == 1.0
    assert out['f1_2'].values[-2] == 1.0
    assert out['f1_2'].values[-1] == 0.0
    assert out['f1_3'].values[3] == 1.0
    assert out['f1_3'].values[0] == 0.0

    proc = PreProcess(A.values)
    np_out = proc.process_one_hot(convert_int=False)
    assert np.allclose(out, np_out)

    proc = PreProcess(A)
    out = proc.process_one_hot(convert_int=True)
    assert 'f3' not in out
    assert (out.iloc[:, 1:5].sum(axis=1) == np.ones(7)).all()
    assert (out.iloc[:, 5:].sum(axis=1) == np.ones(7)).all()


def test_categories():
    """Verify predefined categories handle missing data"""
    proc = PreProcess(A)
    out = proc.process_one_hot(convert_int=False)
    assert (out.columns == ['f2', 'f3', 'f1_0', 'f1_1',
                            'f1_2', 'f1_3']).all()

    # Verify columns are created for missing categories
    # and that the new one-hot columns have names corresponding to their values
    proc = PreProcess(A)
    out0 = proc.process_one_hot(
        convert_int=False, categories={'f1': ['a', 'b', 'c', 'd', 'missing']})
    assert (out0.columns == ['f2', 'f3', 'a', 'b', 'c', 'd', 'missing']).all()
    assert (out0['missing'] == np.zeros(7)).all()

    # verify ordering works.
    out1 = proc.process_one_hot(
        convert_int=False, categories={'f1': ['missing', 'd', 'c', 'a', 'b']})
    assert (out1.columns == ['f2', 'f3', 'missing', 'd', 'c', 'a', 'b']).all()
    assert all(out0.a == out1.a)
    assert all(out0.b == out1.b)
    assert all(out0.c == out1.c)
    assert all(out0.d == out1.d)
    assert (out1['missing'] == np.zeros(7)).all()
    assert out1.a.values[0] == 1
    assert out1.a.values[1] == 0
    assert out1.a.values[2] == 0
    assert out1.a.values[3] == 0
    assert out1.a.values[4] == 1

    # Verify good error with bad categories input.
    try:
        proc.process_one_hot(categories={'f1': ['a', 'b', 'c']})
    except ValueError as e:
        assert 'Found unknown categories' in str(e)
