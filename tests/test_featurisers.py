import numpy as np
import pytest

from chemsci.featurisers import _DEAFULT_FEATURISERS
from chemsci.custom_featurisers import Map4Fingerprint, PubchemFingerprint

from tests.constants import (
    MOL,
    PUB_MOL,
    STD_FEATURES,
    MAP4_FEATURES,
    PUB_FEATURES
)
# ----------------------------------------------------------------------------------------------------------------------


def test_standard_featurisers():
    """Confirm standard rdkit featurisations work."""
    for k in _DEAFULT_FEATURISERS:
        func = _DEAFULT_FEATURISERS[k]
        expected = STD_FEATURES[k]
        out = func(MOL)
        assert np.array_equal(expected, out), 'Reference fingerprints do not match reference values.'

# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("feat, ref_key",
                         [
                             (Map4Fingerprint(), 'map4'),
                             (Map4Fingerprint(is_counted=True), 'is_counted'),
                             (Map4Fingerprint(is_folded=True), 'is_folded')
                         ])
def test_map4_featuriser(feat, ref_key):
    """confirm custom MAP4 featurisation works."""
    out = feat(MOL)
    expected = MAP4_FEATURES[ref_key]
    assert np.array_equal(expected, out), 'Reference fingerprints do not match reference values.'

# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("feat, ref_key",
                         [
                             (PubchemFingerprint('cactvs_fingerprint'), 'cactv'),
                             (PubchemFingerprint('Fingerprint'), 'fingerprint')
                         ])
def test_pubchem_featuriser(feat, ref_key):
    """convert custom pubchem featurisation works."""
    out = feat(PUB_MOL)
    expected = PUB_FEATURES[ref_key]
    assert np.array_equal(expected, out), 'Reference fingerprints do not match reference values.'

# ----------------------------------------------------------------------------------------------------------------------
