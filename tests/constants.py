from rdkit.Chem import MolFromSmiles

SMILES = 'O=C1C(=Cc2ccccc2)CCCC1=Cc1ccccc1'
MOL = MolFromSmiles(SMILES)
assert MOL, 'Error creating the testing mol object.'
