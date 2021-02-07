import pytest

import numpy as np

from rdkit.Chem.rdchem import Mol
from chemsci.factory import FeatureFactory

from tests.constants import (
    SMILES,
    MOL,
    STD_FEATURES,
)

FF_KWARGS = {'converter': 'smiles', 'featuriser': 'maccs'}

# ----------------------------------------------------------------------------------------------------------------------


def test_convert_rep_pass():
    ff = FeatureFactory(**FF_KWARGS)
    out = ff.convert_rep(SMILES)
    assert isinstance(out, Mol), 'Mol object shouldbe returned.'


def test_featurise_mol_pass():
    ff = FeatureFactory(**FF_KWARGS)
    out = ff.featurise_mol(MOL)
    assert np.array_equal(out, STD_FEATURES['maccs']), 'REf maccs fingerprint should be returned.'


def test_convert_featurisation_fail():
    ff = FeatureFactory(**FF_KWARGS)
    out_1 = ff.convert_rep(1234)
    out_2 = ff.featurise_mol(1234)
    assert out_1 is None, 'Invalid rep should return None'
    assert np.array_equal(out_2, np.array([])), 'Should return empty array.'

# ----------------------------------------------------------------------------------------------------------------------


def test_transform():
    """Confirm `fit_transform` method functions as appropriate."""
    ff = FeatureFactory(**FF_KWARGS)
    X = [SMILES] * 10
    out = ff.fit_transform(X)

    assert isinstance(out, list), 'transformation should return list.'
    assert len(out) == len(X), 'Transformation should return as many featurisations as inputs.'
    assert len(ff) == len(X) == len(ff.data), 'Length of factory should be equivalent to length of `data'
    assert out == ff.data, 'Data attribute should be set to transformation results.'

    for feat in out:
        assert np.array_equal(feat, STD_FEATURES['maccs']), 'SMILES should be transformed to featurisation.'

# ----------------------------------------------------------------------------------------------------------------------
