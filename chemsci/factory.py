import warnings

from sklearn.base import TransformerMixin

from chemsci.converters import _DEFAULT_CONVERTERS
from chemsci.featurisers import _DEAFULT_FEATURISERS
from chemsci.exceptions import ConversionWarning, FeaturisationWarning
from chemsci.utils import determine_default_or_callable
# ----------------------------------------------------------------------------------------------------------------------


class FeatureFactory(TransformerMixin):
    """Transformer object to flexibly convert and featurise a variety of molecular representations into a consistent
    format for use in machine learning and other informatics analysis.
    """

    def __init__(self, converter, featuriser):
        """
        Parameters
        ----------
        converter : str OR callable
            Used to convert a molecular representation into a workable object which can undergo featurisation.
            Either default parameter in list or a custom callable object / function, will be used via `__call__`.
                * `smiles`
                * `smarts`
                * `inchi`
                * `mol` (path to file)
                * `pdb` (path to file)
                * 'pubchem' (CID number)
                * `none` (when no conversion is required)

        featuriser : str OR callable
            Used to featurise a workable molecular object.
            Either default parameter in list or a custom callable object / function, will be used via `__call__`.
                * `maccs` (Molecular Access fingerprint)
                * `avalon` (Avalon fingerprint)
                * `daylight` (daylight fingerprint)
                * `ecfp_4_1024` (1024 bit ECFP of radius 4)
                * `ecfp_6_1024` (1024 bit ECFP of radius 4)
                * `ecfp_4_2048` (1024 bit ECFP of radius 4)
                * `ecfp_6_2048` (1024 bit ECFP of radius 4)
                * `fcfp_4_1024` (1024 bit ECFP of radius 4)
                * `fcfp_6_1024` (1024 bit ECFP of radius 4)
                * `fcfp_4_2048` (1024 bit ECFP of radius 4)
                * `fcfp_6_2048` (1024 bit ECFP of radius 4)
                * `pubchem_cactvs` (Presence of 881 substructures from PubChem API)
                * `pubchem_fp` (Encoded fingerpint from PubChem API)
        """
        self.converter = determine_default_or_callable(converter, _DEFAULT_CONVERTERS)
        self.featuriser = determine_default_or_callable(featuriser, _DEAFULT_FEATURISERS)
        self.features = []

    def convert_rep(self, representation):
        """Convert molecular representation into workable molecular object which can be featurised.
        Uses the `converter` callable passed at initialisation.

        Parameters
        ----------
        representation : Any
            String or other representation of a molecular structure / system.

        Raises
        ------
        ConversionWarning : If unable to convert representation into working object.

        Returns
        -------
        mol :
            Workable molecular structure / system which is capable of being featurised.
        """
        try:
            mol = self.converter(representation)
        except:
            warnings.warn(F'Unable to convert repesentation {representation} to workable object.', ConversionWarning)
            mol = None
        return mol

    def featurise_mol(self, mol):
        """Featurise the molecular representation into a numpy array using the `featuriser` passed at initialisation.

        Parameters
        ----------
        mol :
            Workable molecular structure / system which is capable of being featurised.

        Raises
        ------
        FeaturisationWarning : If unable to create featurised representation of passed `mol` object.

        Returns
        -------
        feat : np.ndarray
            Numpy array of featurised molecular system.
        """
        try:
            feat = self.featuriser(mol)
        except:
            warnings.warn(F'Unable to create featurisation of molecular representation {mol}.', FeaturisationWarning)
            feat = None
        return feat

    def tranform(self, X):
        """Apply featurisation to passed iterable of molecular representations.

        Parameters
        ----------
        X : Iterable
            shape (n_entries, )
            Iterable of molecular representations to be converted and then featurised.

        Returns
        -------
        features : list[np.ndarray]
            List of molecular featurisations in the form of numpy arrays.
        """
        features = []
        for representation in X:
            mol = self.convert_rep(representation)
            feat = self.featurise_mol(mol)
            features.append(feat)

        self.features = features
        return self.features

    def fit(self, X, y=None):
        """Included for conformity with sklearn API, otherwise is not used and does not return anything."""
        pass


