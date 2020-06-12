import pandas as pd
from chemsci.fingerprints import MolAccessFF,  DaylightFF, ExtConFingerprintFF, FunctConFingerprintFF

df = pd.read_csv('data/example.csv')
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