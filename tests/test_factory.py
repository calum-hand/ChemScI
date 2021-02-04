import pytest

def test_installation():
    assert True

# check init works fine
# `convert_rep` / `featurise_mol` return as expected / raise correct errors
# `transform`, for given X returns expected array of fingerprints (also attributes are set)
# to<blank>
# to_<file> --> check file and read back to make sure all makes sense
# `__len__` returns correct number (just set arbitrary list size)