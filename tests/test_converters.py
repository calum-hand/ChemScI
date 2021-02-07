"""
No rdkit convertions were tested as already tested as part of `rdkit` library.
"""
import pytest

from chemsci.converters import (
    no_conversion_required,
    pubchem_conv
)

from tests.constants import (
    MOL,
    PUB_MOL
)

# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rep",
                         [
                             MOL,
                             PUB_MOL,
                             'rep',
                             1234
                         ])
def test_no_conv_requied(rep):
    """Confirm inputs are returned.
    """
    assert rep == no_conversion_required(rep), 'Passed representation should be returned.'

# ----------------------------------------------------------------------------------------------------------------------


def test_pubchem_conv():
    """confirm `pubchempy.Compound` object created correctly.
    """
    conv = pubchem_conv()
    rep = conv(5090)
    assert rep == PUB_MOL, 'Expected pubchem Compound object not returned.'

# ----------------------------------------------------------------------------------------------------------------------
