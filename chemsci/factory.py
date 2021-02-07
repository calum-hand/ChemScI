import warnings
import json

import yaml
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from tqdm import tqdm

from chemsci.converters import _DEFAULT_CONVERTERS
from chemsci.featurisers import _DEAFULT_FEATURISERS
from chemsci.exceptions import ConversionWarning, FeaturisationWarning, JsonSerialisationError, FeatureRepresentationError
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
        """
        self.converter = determine_default_or_callable(converter, _DEFAULT_CONVERTERS)
        self.featuriser = determine_default_or_callable(featuriser, _DEAFULT_FEATURISERS)
        self.data = []

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
            An empty numpy array will be returned in this instance to allow the remaining code to still function.

        Returns
        -------
        feat : np.ndarray
            Numpy array of featurised molecular system.
        """
        try:
            feat = self.featuriser(mol)
        except:
            warnings.warn(F'Unable to create featurisation of molecular representation {mol}.', FeaturisationWarning)
            feat = np.array([])
        return feat

    def transform(self, X):
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
        for representation in tqdm(X):
            mol = self.convert_rep(representation)
            feat = self.featurise_mol(mol)
            features.append(feat)

        self.data = features
        return self.data

    def to_list(self):
        """Returns a list of molecule featurisations.

        Returns
        -------
        out : list[np.array]
            shape(num_featurisations, )
            List of featurisations represented as numpy arrays.
        """
        out = self.data
        return out

    def to_dict(self, convert_string=False):
        """Returns a dict of molecule featurisations indexed by integer location in list of featurisations.
        In cases where the featurisation can be represented as a string, then featurisations this can be specified
        through setting `convert_string`.

        Parameters
        ----------
        convert_string : bool
            (default = False)
            Whether featurisations should be presented as strings representations or not.

        Raises
        ------
        FeatureRepresentationError
            If a featurisation canot be converted into a bit string / vector.

        Returns
        -------
        out : dict, {int: Any}
            Dictionary of molecular featurisations keyed by index location in list.
        """
        out = {}
        for ind, f in zip(range(len(self.data)), self.data):
            if convert_string:
                try:
                    out[ind] = str(f)
                except:
                    raise FeatureRepresentationError(F'Unable to represent {ind} as string values in dict.')
            else:
                out[ind] = f
        return out

    def to_array(self, as_type=None):
        """Returns a numpy array of moleculer representations, the type of which can be specified through setting
        `as_type` so long as the representation is valid.

        Parameters
        ----------
        as_type : numpy array type {str, int, float} (default = str)
            Data type which the contents of each numpy array should be returned as.

        Raises
        ------
        FeatureRepresentationError
            If a featueisation canot be converted into the specified array type.

        Returns
        -------
        out : np.array
            shape(num_featurisations, )
            Numpy array of featurisations of given type  `as_type` if specified.
        """
        out = np.array(self.to_list())
        if as_type:
            try:
                out = out.astype(as_type)
            except ValueError:
                raise FeatureRepresentationError(F'Unable to represent array of featurisations as type {as_type}.')
        return out

    def to_series(self, index=None, name=None, as_type=None):
        """Returns a pandas Series of molecular featurisations indexed against integer index.

        Parameters
        ----------
        index : iterable
            (default = None)
            Custom index to be passed to the created pandas series, if `None` is passed then the default pandas integer
            index will be used.

        name : str
            (default = None)
            Name of pandas series to be returned.

        as_type : type
            (default = None)
            Data type which each element in the series should be returned as if possible

        Returns
        -------
        out : pd.Series
            shape(num_featurisations, )
            Pandas Series object containing featurisations, indexed as specified.
        """
        data = self.to_list()
        out = pd.Series(data=data, index=index, dtype=as_type, name=name)
        return out

    def to_csv(self, path, index=None, name=None, as_type=None, sep=',', save_index=True, columns=None, header=True):
        """Convenience function to save a pandas series of featurisations to csv file.
        Equivalent to calling:
        >>> ...
        >>> ff.to_series()
        >>> ff.to_csv('path/to/file.csv')

        Parameters
        ----------
        path : str
            Path to location to save featurisations to (calls pandas `to_csv`).

        index : iterable
            (default = None)
            Custom index to be passed to the created pandas series, if `None` is passed then the default pandas integer
            index will be used.

        name : str
            (default = None)
            Name of pandas series to be returned.

        as_type : type
            (default = None)
            Data type which each element in the series should be returned as if possible

        sep : str
            (default = ',')
            Delimiter to use in the csv file.

        save_index : bool
            (default = True)
            Specifies if index values should be saved to the csv file.

        columns : list[str]
            (default = None)
            Column headers to be used in the csv file.

        header : bool
            (default = True)
            Specifies if column headers should be written to the csv file.

        Returns
        -------
        None
        """
        out = self.to_series(index=index, name=name, as_type=as_type)
        path = str(path)
        out.to_csv(path_or_buf=path, sep=sep, index=save_index, columns=columns, header=header)

    def to_json(self, path=None):
        """Returns a JSON string of molecule featurisations or saves to specified file.
        If `path` is specified, then nothing will be returned.

        Parameters
        ----------
        path : str
            Path to location to save featurisations to.

        Raises
        ------
        JsonSerialisationError:
            If unable to serialise the representation and featurisations to JSON string.

        Returns
        -------
        out : str {str: str}
            JSON string of molecular featurisations.
            if `path` then nothing is returned.
        """
        dict_rep = self.to_dict(convert_string=True)
        try:
            out = json.dumps(dict_rep)
        except TypeError:
            raise JsonSerialisationError('Unable to serialise features data to JSON format.')
        if path is None:
            return out
        else:
            with (path, 'w') as f:
                json.dump(out, f, indent=4)

    def to_yaml(self, path):
        """Saves the molecule featurisations to a YAML file.

        Parameters
        ----------
        path : str
            Path to location to save featurisations to.

        Returns
        -------
        None
        """
        dict_rep = self.to_dict(convert_string=True)
        path = str(path)
        with open(path, 'w') as f:
            yaml.dump(dict_rep, f)

    def fit(self, X, y=None):
        """Included for conformity with sklearn API, otherwise is not used and does not return anything."""
        return self

    def __len__(self):
        """Number of featurisations.
        """
        return len(self.data)

    def __repr__(self):
        """String representation of `FeatureFactory` instance."""
        out = F'Factory: Converter={self.converter}, Featuriser={self.featuriser}.'
        return out
