import numpy as np

from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

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


class DaylightFingerprint:
    """Generates the Daylight fingerprint for a passed `rdkit.Chem.rdchem.Mol' object using
    `rdkit.Chem.rdmolops.RDKFingerprint`.
    """

    def __init__(self, nbits=2048, min_path=1, max_path=7):
        """
        Parameters
        ----------
        nbits : int
            (default = 2048)
            Number of bits in the output fingerprint.

        min_path : int
            (default = 1)
            Minimum number of bonds to include in subgraph calculation.

        max_path: int
            (default = 7)
            Maximum number of bonds to include in subgraph calculation.
        """
        self.nbits = int(nbits)
        self.min_path = int(min_path)
        self.max_path = int(max_path)

    def __call__(self, mol):
        """Generates the Daylight fingerprint for passed `mol` object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Rdkit mol object.

        Returns
        -------
        fp_arr : np.ndarray, shape(self.nbits, )
            Fingerprint expressed as a numpy row vector.
        """
        fp = RDKFingerprint(mol, fpSize=self.nbits, minPath=self.min_path, maxPath=self.max_path)
        fp_arr = _rdkit_fp_to_np_arr(fp)
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


class MorganFingerprint:
    """Generates variations of morgan based fingerprints (including `ECFP` and `FCFP`) for a passed
    `rdkit.Chem.rdchem.Mol` object.

    Uses `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect` to generate the fingerprints.
    """

    def __init__(self, nbits=1024, diameter=4, use_features=False):
        """
        Parameters
        ----------
        nbits : int
            (default = 1024)
            Number  of bits in the output fingerprint.

        diameter : int
            (default = 4)
            Number of bonds to include in circular fingerprint.

        use_features : bool
            (default = False)
            Denotes if the ECFP (False) or FCFP (True) should be generated.
        """
        self.nbits = int(nbits)
        self.diameter = int(diameter)
        self.use_features = bool(use_features)
        self._radius = self.diameter // 2

    def __call__(self, mol):
        """Generates the Morgan fingerprint for passed `mol` object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Rdkit mol object.

        Returns
        -------
        fp_arr : np.ndarray, shape(self.nbits, )
            Fingerprint expressed as a numpy row vector.
        """
        fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self.nbits, useFeatures=self.use_features)
        fp_arr = _rdkit_fp_to_np_arr(fp)
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------


_DEAFULT_FEATURISERS = {'maccs': maccs_fp,
                        'avalon': avalon_fp,
                        'daylight': DaylightFingerprint(),
                        'ecfp_4_1024': MorganFingerprint(nbits=1024, diameter=4),
                        'ecfp_6_1024': MorganFingerprint(nbits=1024, diameter=6),
                        'ecfp_4_2048': MorganFingerprint(nbits=2048, diameter=4),
                        'ecfp_6_2048': MorganFingerprint(nbits=2048, diameter=6),
                        'fcfp_4_1024': MorganFingerprint(nbits=1024, diameter=4, use_features=True),
                        'fcfp_6_1024': MorganFingerprint(nbits=1024, diameter=6, use_features=True),
                        'fcfp_4_2048': MorganFingerprint(nbits=2048, diameter=4, use_features=True),
                        'fcfp_6_2048': MorganFingerprint(nbits=2048, diameter=6, use_features=True)
                        }

# ----------------------------------------------------------------------------------------------------------------------
