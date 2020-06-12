import time

from pubchempy import Compound
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint

from _base.fingerprint import FingerprintFactory

# ----------------------------------------------------------------------------------------------------------------------


class MolAccessFF(FingerprintFactory):
    def __init__(self):
        super().__init__()
        self.nbits = 166

    def mol_to_fingerprint(self, mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_bit = fp.ToBitString()
        cleaned_fp = list(fp_bit)[1:]
        return cleaned_fp


# ----------------------------------------------------------------------------------------------------------------------


class DaylightFF(FingerprintFactory):
    def __init__(self, nbits=2048, min_path=1, max_path=7):
        super().__init__()
        self.nbits = nbits
        self.min_path = min_path
        self.max_path = max_path

    def mol_to_fingerprint(self, mol):
        fp = RDKFingerprint(mol, fpSize=self.nbits, minPath=self.min_path, maxPath=self.max_path)
        fp_bit = fp.ToBitString()
        cleaned_fp = list(fp_bit)
        return cleaned_fp


# ----------------------------------------------------------------------------------------------------------------------


class ExtConFingerprintFF(FingerprintFactory):

    _features = False

    def __init__(self, nbits=1024, diameter=4):
        super().__init__()
        self.nbits = nbits
        self.diameter = diameter
        self._radius = self.diameter // 2

    def mol_to_fingerprint(self, mol):
        fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self.nbits, useFeatures=self._features)
        fp_bit = fp.ToBitString()
        cleaned_fp = list(fp_bit)
        return cleaned_fp


# ----------------------------------------------------------------------------------------------------------------------


class FunctConFingerprintFF(ExtConFingerprintFF):
    _features = True


# ----------------------------------------------------------------------------------------------------------------------

class PubChemFF(FingerprintFactory):

    representation_converter = Compound.from_cid
    # TODO : Implement the crawl delay for `representation_to_mol` in this case
    # TODO : Will have to overload `representation_to_mol` so can include crawl delay and allow multiprocessing.

    def __init__(self, crawl_delay=2):
        super().__init__()
        self.nbits = 881
        self._crawl_delay = crawl_delay

    def mol_to_fingerprint(self, mol):
        fp_bit = mol.cactvs_fingerprint
        cleaned_fp = list(fp_bit)
        return cleaned_fp
