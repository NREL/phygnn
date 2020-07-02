"""
Test the pre processing module.
"""
import numpy as np
import pandas as pd
from phygnn.pre_processing import PreProcess


index = pd.date_range('20180101', '20190101', freq='5min')
A = pd.DataFrame({'f1': ['a', 'b', 'c', 'd', 'a', 'c', 'a'],
                  'f2': np.arange(7) * 0.333,
                  'f3': np.arange(7)}, index=index[:7])


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
