import numpy as np

from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from chemsci.exceptions import UserSelectionError

# ----------------------------------------------------------------------------------------------------------------------


def _rdkit_fp_to_np_arr(fp):
    """Convert an `rdkit` molecular fingerprint into a 1D `numpy` row vector.
    First takes the passed fingeprint to extract the bit string which is then converted to an array of size `n` where
    each entry in the fingerprint is a separate element in the array.

    Parameters
    ----------
    fp : molecular fingerprint generated through `rdkit`

    Returns
    -------
    fp_arr : np.ndarray
        Row vector of passed fingerprint.
    """
    fp_bit = fp.ToBitString()
    fp_arr = np.array(list(fp_bit))
    return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


def maccs_fp(mol):
    """Generates the Molecular Access Fingerprints (MACCS) for a passed 'rdkit.Chem.rdchem.Mol' object

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        `rdkit` mol object.

    Notes
    -----
    As the `rdkit` implementation of `rdkit.Chem.MACCSkeys.GenMACCSKeys` generates a 167 bit vector, here the dead bit
    at index 0 is removed to ensure the resulting fingerprint is the correct length.

    Returns
    -------
    fp_arr : np.ndarray, shape(166,)
        Fingerprint expressed as a numpy row vector.
    """
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_arr = _rdkit_fp_to_np_arr(fp)[1:]  # index to remove dead bit
    return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


def avalon_fp(mol):
    """Generates the Avalon fingerprint for a passed 'rdkit.Chem.rdchem.Mol' object using
    `rdkit.Avalon.pyAvalonTools.GetAvalonFP`.

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
    fp_arr = _rdkit_fp_to_np_arr(fp)
    return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class daylight_fp:
    """

    """

    def __init__(self, nbits=2048, min_path=1, max_path=7):
        """
        Parameters
        ----------
        nbits
        min_path
        max_path
        """
        self.nbits = nbits
        self.min_path = min_path
        self.max_path = max_path

    def __call__(self, mol):
        """

        Parameters
        ----------
        mol

        Returns
        -------

        """
        fp = RDKFingerprint(mol, fpSize=self.nbits, minPath=self.min_path, maxPath=self.max_path)
        fp_arr = _rdkit_fp_to_np_arr(fp)
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class morgan_ecfp:

    _features = False

    def __init__(self, nbits=1024, diameter=4):
        self.nbits = nbits
        self.diameter = diameter
        self._radius = self.diameter // 2

    def __call__(self, mol):
        fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self.nbits, useFeatures=self._features)
        fp_arr = _rdkit_fp_to_np_arr(fp)
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class morgan_fcfp(morgan_ecfp):
    _features = True

# ----------------------------------------------------------------------------------------------------------------------


class pubchem_fp:
    valid_fingerprints = ['cactvs_fingerprint', 'fingerprint']

    def __init__(self, pub_fp='cactv_fingerprint'):
        self.pub_fp = str(pub_fp).lower()

        if self.pub_fp not in self.valid_fingerprints:
            raise UserSelectionError(F'Passed {self.pub_fp} not in {self.valid_fingerprints}.')

    def __call__(self, mol):
        if self.pub_fp == self.valid_fingerprints[0]:
            fp_bit = mol.cactvs_fingerprint  # attribute for Compound object in `PubchemPy`
        elif self.pub_fp == self.valid_fingerprints[1]:
            fp_bit = mol.fingerprint
        else:
            raise AttributeError(F'Incorrect fingerprint specified. {self.pub_fp} not supported by PubChemPy API.')
        fp_arr = np.array(list(fp_bit))
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------
