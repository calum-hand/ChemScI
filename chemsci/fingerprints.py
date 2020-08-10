import numpy as np

from pubchempy import Compound
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from chemsci.base.fingerprint import FingerprintFactory

# ----------------------------------------------------------------------------------------------------------------------


class MolAccessFF(FingerprintFactory):
    def __init__(self):
        """Fingerprint Factory for obtaining Molecular Access Fingerprints (MACCS).
        Implementation uses `rdkit.Chem.MACCSkeys.GenMACCSKeys` to obtain fingerprint.
        Inherets from `FingerprintFactory.

        Notes
        -----
        As the `rdkit` implementation of `GenMACCSKeys` generates a 167 bit vector, here the

        Attributes
        ----------
        nbits : int (default = 166)
            Number of bits present in the MACCS fingerprint.
            Ths number is the standard value for MACCS fingerprints and should not be altered.
        """
        super().__init__()
        self.nbits = 166

    def mol_to_fingerprint(self, mol):
        """Generates the Molecular Access Fingerprints (MACCS) for a passed 'rdkit.Chem.rdchem.Mol' object.
        As the `rdkit` implemented algorithm produces a 167 bit vector, here the returned vector does not include
        the bit value at index `0` as this is a "dead" bit.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            `rdkit` mol object.

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


class AvalonFF(FingerprintFactory):
    """Fingerprint Factory for obtaining Avalon Fingerprints.
    Implementation uses `rdkit.Avalon.pyAvalonTools.GetAvalonFP` to obtain fingerprint.
    Inherets from `FingerprintFactory.

    Attributes
    ----------
    nbits : int (default = 512)
        Number of bits present in the Avalon fingerprint.
        Ths number is the standard value for Avalon fingerprints and should not be altered.
    """
    def __init__(self):
        super().__init__()
        self.nbits = 512

    def mol_to_fingerprint(self, mol):
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


class DaylightFF(FingerprintFactory):
    def __init__(self, nbits=2048, min_path=1, max_path=7):
        super().__init__()
        self.nbits = nbits
        self.min_path = min_path
        self.max_path = max_path

    def mol_to_fingerprint(self, mol):
        fp = RDKFingerprint(mol, fpSize=self.nbits, minPath=self.min_path, maxPath=self.max_path)
        fp_bit = fp.ToBitString()
        fp_arr = np.array(list(fp_bit))
        return fp_arr


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
        fp_arr = np.array(list(fp_bit))
        return fp_arr


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
        fp_arr = np.array(list(fp_bit))
        return fp_arr
