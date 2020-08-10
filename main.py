import pandas as pd
from chemsci.fingerprints import MolAccessFF,  DaylightFF, ExtConFingerprintFF, FunctConFingerprintFF

df = pd.read_csv('data/molecules.csv')
smiles = df['smiles']

macc_ff = MolAccessFF()
macc_ff.obtain_representations(smiles)
macc_ff.obtain_fingerprints()

day_ff = DaylightFF(max_path=4)
day_ff.obtain_representations(smiles)
day_ff.obtain_fingerprints()

ecfp_ff = ExtConFingerprintFF()
ecfp_ff.obtain_representations(smiles)
ecfp_ff.obtain_fingerprints()

fcfp_ff = FunctConFingerprintFF()
fcfp_ff.obtain_representations(smiles)
fcfp_ff.obtain_fingerprints()

# TODO : 0) Make more abstract (i.e. factory is one object that is inherited from)
# TODO : +) Make class method for features to load from a chemical file into the relevant feature format)
#           i.e. class MolAccessFF(Factory, Loader, Transformer):
# TODO : 1) Add more formats for output of features (YAML, SQLITE3)
# TODO : 2) Add option to return as string if not file name is passed
# ~~TODO : 3) Make like transformer for easy integration into sklearn~~
# TODO : 4) Add more fingerprints from literature (if code available and suitable license exists)
# TODO : 5) Add other features like coloumb matrix and Ewalld summation matrix 
# Aim is to be a lightweight tool for researchers to easily add their own features