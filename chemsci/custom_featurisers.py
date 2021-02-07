import itertools
from collections import defaultdict

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix

from mhfp.encoder import MHFPEncoder
import tmap as tm

from chemsci.exceptions import UserSelectionError

# ----------------------------------------------------------------------------------------------------------------------


class Map4Fingerprint:
    """Calculates the atom pair minmashed fingerprint for a given molecular object.
    Fingerprint is as described by `DOI: 10.1186/1758-2946-5-26` and implemented in the
    [corresponding repository](https://github.com/reymond-group/map4).
    """

    def __init__(self, dimensions=1024, radius=2, is_counted=False, is_folded=False, return_strings=False):
        """
        Parameters
        ----------
        dimensions : int
            (default = 1024)
            Number of entries in the output map4 fingerprint.

        radius : int
            (default = 2)
            Number of bonds away from atom centre to consider.

        is_counted : bool
            (default = False)

        is_folded : bool
            (default = False)

        return_strings : bool
            (default = False)
            If True then returns substructure strings rather than hashed fingerprint.
        """
        self.dimensions = int(dimensions)
        self.radius = int(radius)
        self.is_counted = bool(is_counted)
        self.is_folded = bool(is_folded)
        self.return_strings = bool(return_strings)

        if self.is_folded:
            self.encoder = MHFPEncoder(dimensions)
        else:
            self.encoder = tm.Minhash(dimensions)

    def __call__(self, mol):
        """Calculates the atom pair minmashed fingerprint for a given molecular object.
        Fingerprint is as described by `DOI: 10.1186/1758-2946-5-26` and implemented in the
        [corresponding repository](https://github.com/reymond-group/map4).

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            `rdkit` mol object.

        Returns
        -------
        fp_arr : np.ndarray
            shape(self.dimensions, )
            Map4 fingerprint.
        """
        atom_envs = self._get_atom_envs(mol)
        atom_env_pairs = self._all_pairs(mol, atom_envs)
        if self.is_folded:
            fp_arr = self._fold(atom_env_pairs)
        elif self.return_strings:
            fp_arr = atom_env_pairs
        else:
            fp_arr = self.encoder.from_string_array(atom_env_pairs)
        return np.asarray(fp_arr)

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)

    def _get_atom_envs(self, mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []
                atoms_env[idx].append(Map4Fingerprint._find_env(mol, idx, radius))
        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles
        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))
        return list(set(atom_pairs))

# ----------------------------------------------------------------------------------------------------------------------


class PubchemFingerprint:
    """Featuriser used to interact with generated `pubchempy.Compound` objects and retrieve relevant fingerprint records
    / data.
    """
    valid_fingerprints = ['cactvs_fingerprint', 'fingerprint']

    def __init__(self, pub_fp='cactv_fingerprint'):
        """
        Parameters
        ----------
        pub_fp : str
            The specific PubChem fingerprint to be retrieved from the pubchempy.Compound object.
            Can either be 'cactvs_fingerprint' OR 'fingerprint'.
        """
        self.pub_fp = str(pub_fp).lower()

        if self.pub_fp not in self.valid_fingerprints:
            raise UserSelectionError(F'Passed {self.pub_fp} not in {self.valid_fingerprints}.')

    def __call__(self, mol):
        """Retrieves the specified PubChem fingerprint for passed `mol` object.

        Parameters
        ----------
        mol : pubchempy.Compound
            PubChempy Compound object.

        Returns
        -------
        fp_arr : np.ndarray, shape(self.nbits, )
            Fingerprint expressed as a numpy row vector.
        """
        if self.pub_fp == self.valid_fingerprints[0]:
            fp_bit = mol.cactvs_fingerprint  # attribute for Compound object in `PubchemPy`
        elif self.pub_fp == self.valid_fingerprints[1]:
            fp_bit = mol.fingerprint
        else:
            raise AttributeError(F'Incorrect fingerprint specified. {self.pub_fp} not supported by PubChemPy API.')
        fp_arr = np.array(list(fp_bit))
        return fp_arr

# ----------------------------------------------------------------------------------------------------------------------
