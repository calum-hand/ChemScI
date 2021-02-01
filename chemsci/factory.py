import warnings

from sklearn.base import TransformerMixin

from chemsci.exceptions import ConversionWarning, FeaturisationWarning

# ----------------------------------------------------------------------------------------------------------------------


class FeatureFactory(TransformerMixin):
    """

    """

    def __init__(self, converter, featuriser, store_mols=False):
        """
        Parameters
        ----------
        converter : callable
            Instantiaed callable object or function.
            Used to convert a molecular representation into a workable object which can undergo featurisation.

        featuriser : callable
            Instantiaed callable object or function.
            Used to featurise a workable molecular object.

        store_mols : bool (default=False)
            Allows user to specify if they wish for the interim workable molecular objects to be also stored in memory.
            The expectation is that these objects are likely to be large and hence if a significant volume is processed
            could resolve in memory issues, hence default is not to store them.
        """
        assert callable(converter) and callable(featuriser), 'Passed converter and featuriser should be callables'
        self.converter = converter
        self.featuriser = featuriser
        self.features = []
        self.mols = []

        self._store_mols = bool(store_mols)  # dont store by default as potentially large objects.

    def convert_rep(self, representation):  # warning so user knows theres an issue but also doesnt throw a hissy fit
        """

        Parameters
        ----------
        representation

        Returns
        -------

        """
        try:
            mol = self.converter(representation)
        except:
            warnings.warn(F'Unable to convert repesentation {representation} to workable object.', ConversionWarning)
            mol = None
        return mol

    def featurise_mol(self, mol):  # warning as above, passes None so at least there will be consistent output so you can tell which one messed up based on index positoion (maybe)
        """

        Parameters
        ----------
        mol

        Returns
        -------

        """
        try:
            feat = self.featuriser(mol)
        except:
            warnings.warn(F'Unable to create featurisation of molecular representation {mol}.', FeaturisationWarning)
            feat = None
        return feat

    def fit(self, X, y=None):
        """Included for conformity with sklearn API, otherwise is not used and does not return anything."""
        pass

    def tranform(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        features, mols = [], []
        for representation in X:
            mol = self.convert_rep(representation)
            feat = self.featurise_mol(mol)
            features.append(feat)
            if self._store_mols:
                mols.append(mol)

        self.features = features
        self.mols = mols
        return self.features
