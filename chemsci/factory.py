import warnings
import json

import yaml
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from chemsci.converters import _DEFAULT_CONVERTERS
from chemsci.featurisers import _DEAFULT_FEATURISERS
from chemsci.exceptions import ConversionWarning, FeaturisationWarning, JsonSerialisationError
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

        self.data = features
        return self.data

    def fit(self, X, y=None):
        """Included for conformity with sklearn API, otherwise is not used and does not return anything."""
        pass

    def to_list(self):
        """Returns a list of molecule fingerprints.
        By default, will return each fingerprint represented as a numpy array, generated by `mol_to_fingerprint'.
        In cases where the fingerprint can be represented as a bit vector / string, then a list of fingerprint strings
        can be returned instead by setting `as_bit_string` to `True`.

        >>> FingerprintFactory.produce_list(as_bit_string=False)
        [np.array(['1', '1', '0', '0', '1']), ..., np.array(['1','0', '0', '0', '1'])]

        >>> FingerprintFactory.produce_list(as_bit_string=True)
        ['11001', ..., '10001']

        Parameters
        ----------
        as_bit_string : bool (default = False)
            Whether fingerprints should be returned as bit strings or not.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        Returns
        -------
        product : list[np.array] shape(num_fingerprints, fingerprint_length) OR list[str], shape(num_fingerprints)
            List of fingerprints, either represented as numpy arrays or strings.
        """
        out = self.data
        return out

    def to_dict(self, convert_string=False):
        """Returns a dict of molecule representations (assumed capable of being converted to string)
         and corresponding fingerprints of form:

            {representation_0: fingerprint_0, ..., representation_n: fingerprint_n}

        By default, fingerprints are returned as numpy array representations generated by `mol_to_fingerprint'.
        In cases where the fingerprint can be represented as a bit vector / string, then fingerprint strings
        can be returned instead by setting `as_bit_string` to `True`.

        >>> FingerprintFactory.produce_dict(as_bit_string=False)
        {'smiles_0': np.array(['1', '1', '0', '0', '1']), ..., 'smiles_n': np.array(['1','0', '0', '0', '1'])}

        >>> FingerprintFactory.produce_dict(as_bit_string=True)
        {'smiles_0': '11001', ..., 'smiles_n': '10001'}

        Parameters
        ----------
        as_bit_string : bool (default = False)
            Whether fingerprints should be returned as bit strings or not.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        Returns
        -------
        product : dict, {representation: fingerprint}, {str: np.array} OR {str: str}
            Dictionary of molecular representations and corresponding fingerprint.
            Key values (representations) will be converted to strings at evaluation.
        """
        out = {}
        for ind, f in zip(range(len(self.data)), self.data):
            if convert_string:
                out[ind] = str(f)
            else:
                out[ind] = f
        return out

    def to_array(self, as_type=None):
        """Returns a numpy array of molecule fingerprints, the type of which can be specified if
        `as_bit_string=False`.

        >>> FingerprintFactory.produce_array(as_bit_string=False, as_type=int)
        array([[1, 1, 0, 0, 1],
               ...,
               [1, 0, 0, 0, 1]])

        >>> FingerprintFactory.produce_array(as_bit_string=True)
        array(['11001',
               ...,
               '10001'])

        Parameters
        ----------
        as_bit_string : bool (default = False)
            Whether fingerprints should be returned as bit strings or not.

        as_type : numpy array type {str, int, float} (default = str)
            Data type which the contents of each numpy array should be returned as.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        FingerprintRepresentationError
            If unable to convert the output array to the desired type.
            Typically would occur when all obtained fingerprints are not the same length.

        Returns
        -------
        product : np.array, shape(num_fingerprints, fingerprint_length)
            Numpy array of fingerprints where each row is a fingerprint and each column is a specific bit.
        """

        out = np.array(self.to_list())
        if as_type:
            out = out.astype(as_type)
        return out

    def to_series(self):
        """Returns a pandas Series of molecular fingerprints indexed against the correpsonding representation (as str)
        where each element is either a bit string or a `n` dimensional numpy array.
        The ability to store multi-dimensional numpy arrays as entries enables fingerprint representations such as 2D
        correlation matrices or other user implemented fingerprints.

        >>> FingerprintFactory.produce_series(as_bit_string=False, as_type=int)
        'smiles_0'  [1, 1, 0, 0, 1],
         ...        ...
         'smiles_n' [1, 0, 0, 0, 1]
         dtype: object

         >>> FingerprintFactory.produce_series(as_bit_string=True)
        'smiles_0'  '11001',
         ...        ...
         'smiles_n' '10001'
         dtype: object

        Parameters
        ----------
        as_bit_string : bool (default = False)
            Whether fingerprints should be returned as bit strings or not.

        as_type : numpy array type {str, int, float} (default = str)
            Data type which each numpy array element in the series should be returned as.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        FingerprintRepresentationError
            If unable to convert the output array to the desired type.
            Typically would occur when all obtained fingerprints are not the same length.

        Returns
        -------
        product : pd.Series, shape(num_fingerprints)
            Pandas Series object containing either obtained fingerprints as either bit strings or numpy arrays.
            Series is indexed by string molecular representations.
        """
        out = pd.Series(data=self.to_list())
        return out

    def to_csv(self, path, sep=',', index=True, columns=None, header=True):
        out = self.to_series()
        out.to_csv(path_or_buf=path, sep=sep, index=index, columns=columns, header=header)

    def to_json(self, path=None):
        """Returns a JSON string of molecule representations and corresponding fingerprints.
        By default (`as_bit_string=False`), fingerprints are converted from their numpy arrays to nested lists
        for serialisation to JSON (multi dimensional fingerprints are supported).
        Otherwise the bit string fingerprints are serialised (if possible).

        >>> FingerprintFactory.produce_json(as_bit_string=False)
        '{"smiles_0": ["1", "1", "0", "0", "1"], ..., "smiles_n": ["1", "0", "0", "0", "1"]}'

        >>> FingerprintFactory.produce_json(as_bit_string=True)
        '{"smiles_0": "11001", ..., "smiles_n": "10001"}'

        Parameters
        ----------
        as_bit_string : bool (default = False)
            Whether fingerprints should be returned as bit strings or not.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        JsonSerialisationError
            If unable to serialise the representation and fingerprints to JSON string.

        Returns
        -------
        product : str, {representation: fingerprint}, {str: list[str]} OR {str: str}
            JSON string of molecular representations and corresponding fingerprint.
            Fingerprint can either be a serialised nested list or string.
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
                json.dump(out, f)

    def to_yaml(self, path):
        dict_rep = self.to_dict(convert_string=True)
        with open(path, 'w') as f:
            yaml.dump(dict_rep, f)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        out = F'Factory: Converter={self.converter}, Featuriser={self.featuriser}.'
        return out

