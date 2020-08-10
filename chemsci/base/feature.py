from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin
from rdkit.Chem import MolFromSmiles, MolFromInchi, MolFromSmarts

from chemsci.base.exceptions import StandardRepresentationError

# ----------------------------------------------------------------------------------------------------------------------


class Feature(ABC):
    """Lays out requirements for a feature:
        1. Must be capable of being converted from representation into actionable form (i.e. SMILES --> `rdkit.Mol`)
        2. Must be able to action said form into defined feature (i.e. `rdkit.Mol` --> `np.ndarray`)
    """
    @abstractmethod
    def convert_representation(self, representation):
        """Convert molecular representation / file into aobject capable of being calculated on to generate feature.
        Must accept single representation as input and return a singular correpsonding mol like object.

        Parameters
        ----------
        representation : str
            Smiles string representation of olecule to be converted to mol object.

        Returns
        -------
        mol : `rdkit.Chem.rdchem.Mol`
            Singular mol object as implemented by `rdkit`.
        """
        return NotImplemented

    @abstractmethod
    def generate_feature(self, mol):
        """Calculates and returns the fingerprint of a singular mol object as a numpy array.

        Parameters
        ----------
        mol : `rdkit.Chem.rdchem.Mol`
            Singular mol object as implemented by `rdkit`.

        Notes
        -----
        The returned numpy array **must** be a row vector if the fingerprint is 1D where each element is a specific bit.
        If the fingerprint is multi-dimensional then the `n`D vector can be returned without constraint.

        Returns
        -------
        feature : np.ndarray
        """
        return NotImplemented


# ----------------------------------------------------------------------------------------------------------------------


class CustomFeatureTransformer(Feature, TransformerMixin):
    """Allows for interface with `sklearn` objects through definition of `fit` and `transform` methods.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for representation in X:
            mol = self.convert_representation(representation)
            feature = self.generate_feature(mol)
            features.append(feature)
        return features


# ----------------------------------------------------------------------------------------------------------------------


class StandardFeatureTransformer(CustomFeatureTransformer):
    """Assumes user will be converting from representation to actionable mol object via standard methods:
        * SMILES conversion
        * InChi conversion
        * Smarts conversion
        * ...
    """

    converters = {'smiles': MolFromSmiles,
                  'inchi': MolFromInchi,
                  'smarts': MolFromSmarts}

    def __init__(self, representation):
        try:
            self.rep_converter = self.converters[representation.lower()]
        except KeyError:
            raise StandardRepresentationError(F'Representation {representation} not in {self.converters.keys()}')

    def convert_representation(self, representation):
        mol = self.rep_converter(representation)
        return mol

# ----------------------------------------------------------------------------------------------------------------------
