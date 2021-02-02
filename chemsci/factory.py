import warnings

from sklearn.base import TransformerMixin

from chemsci.exceptions import ConversionWarning, FeaturisationWarning

# ----------------------------------------------------------------------------------------------------------------------


class FeatureFactory(TransformerMixin):
    """Transformer object to flexibly convert and featurise a variety of molecular representations into a consistent
    format for use in machine learning and other informatics analysis.
    """

    def __init__(self, converter, featuriser):
        """
        Parameters
        ----------
        converter : callable
            Instantiaed callable object or function.
            Used to convert a molecular representation into a workable object which can undergo featurisation.

        featuriser : callable
            Instantiaed callable object or function.
            Used to featurise a workable molecular object.
        """
        assert callable(converter) and callable(featuriser), 'Passed converter and featuriser should be callables'
        self.converter = converter
        self.featuriser = featuriser
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


