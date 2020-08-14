import time

import numpy as np

from pubchempy import Compound
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from chemsci.base.feature import StandardFeatureTransformer, CustomFeatureTransformer

# ----------------------------------------------------------------------------------------------------------------------


class MolAccess(StandardFeatureTransformer):
    """Fingerprint Factory for obtaining Molecular Access Fingerprints (MACCS).
    Implementation uses `rdkit.Chem.MACCSkeys.GenMACCSKeys` to obtain fingerprint.
    """
    def generate_feature(self, mol):
        """Generates the Molecular Access Fingerprints (MACCS) for a passed 'rdkit.Chem.rdchem.Mol' object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            `rdkit` mol object.

        Notes
        -----
        As the `rdkit` implementation of `GenMACCSKeys` generates a 167 bit vector, here the dead bit at index 0
        is removed to ensure the resulting fingerprint is the correct length

        Returns
        -------
        fp_arr : np.ndarray, shape(166,)
            Fingerprint expressed as a numpy row vector.
        """
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_bit = fp.ToBitString()
        fp_arr = np.array(list(fp_bit))[1:]  # index to remove dead bit
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class Avalon(StandardFeatureTransformer):
    """Fingerprint Factory for obtaining Avalon Fingerprints.
    Implementation uses `rdkit.Avalon.pyAvalonTools.GetAvalonFP` to obtain fingerprint.
    """
    def generate_feature(self, mol):
        """Generates the Avalon fingerprint for a passed 'rdkit.Chem.rdchem.Mol' object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            `rdkit` mol object.

        Returns
        -------
        fp_arr : np.ndarray, shape(512,)
            Fingerprint expressed as a numpy row vector.
        """
        fp = GetAvalonFP(mol)
        fp_bit = fp.ToBitString()
        fp_arr = np.array(list(fp_bit))
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class Daylight(StandardFeatureTransformer):
    """
    """

    def __init__(self, representation, nbits=2048, min_path=1, max_path=7):
        super().__init__(representation)
        self.nbits = nbits
        self.min_path = min_path
        self.max_path = max_path

    def generate_feature(self, mol):
        fp = RDKFingerprint(mol, fpSize=self.nbits, minPath=self.min_path, maxPath=self.max_path)
        fp_bit = fp.ToBitString()
        fp_arr = np.array(list(fp_bit))
        return fp_arr


# ----------------------------------------------------------------------------------------------------------------------


class ECFP(StandardFeatureTransformer):

    _features = False

    def __init__(self, representation, nbits=1024, diameter=4):
        super().__init__(representation)
        self.nbits = nbits
        self.diameter = diameter
        self._radius = self.diameter // 2

    def generate_feature(self, mol):
        fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self.nbits, useFeatures=self._features)
        fp_bit = fp.ToBitString()
        fp_arr = np.array(list(fp_bit))
        return fp_arr


# ----------------------------------------------------------------------------------------------------------------------


class FCFP(ECFP):
    _features = True


# ----------------------------------------------------------------------------------------------------------------------

class PubChem(CustomFeatureTransformer):

    def __init__(self, crawl_delay=2):
        self.crawl_delay = crawl_delay

    def convert_representation(self, representation):
        compound = Compound.from_cid(representation)  # calls PubChem API
        time.sleep(self.crawl_delay)
        return compound

    def generate_feature(self, mol):
        fp_bit = mol.cactvs_fingerprint  # attribute for Compound object in `PubchemPy`
        fp_arr = np.array(list(fp_bit))
        return fp_arr
