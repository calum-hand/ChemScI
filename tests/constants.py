import pickle
from rdkit.Chem import MolFromSmiles
from pubchempy import Compound

# ----------------------------------------------------------------------------------------------------------------------


with open('tests/files/features.pkl', 'rb') as f:
    REF_FEATURES = pickle.load(f)

    STD_FEATURES = REF_FEATURES['standard']
    MAP4_FEATURES = REF_FEATURES['map4']
    PUB_FEATURES = REF_FEATURES['pubchem']

# ----------------------------------------------------------------------------------------------------------------------


SMILES = 'O=C1C(=Cc2ccccc2)CCCC1=Cc1ccccc1'
MOL = MolFromSmiles(SMILES)
assert MOL, 'Error creating the testing mol object.'

# ----------------------------------------------------------------------------------------------------------------------


cid = 5090
PUB_MOL = Compound.from_cid(cid)
assert isinstance(PUB_MOL, Compound), 'Error creating the testing pubchem Compound object.'

# ----------------------------------------------------------------------------------------------------------------------
