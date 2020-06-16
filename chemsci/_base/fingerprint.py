from warnings import warn
import json

import pandas as pd
import numpy as np

from rdkit import Chem

from chemsci._exceptions import (BitVectorRepresentationError,
                                 FingerprintRepresentationError,
                                 JsonSerialisationError)


class FingerprintFactory:
    """Parent class used to calculate various molecular fingerprints and return them to the user in a variety of ways.
    Expects all molecular fingerprints of the same type to be equal dimensions regardless of molecule.

    Attributes
    ----------
    _representations : list, shape(num_valid_mol_objects), initialise as `[]`.
        List of molecular representations from which to generate mol objects.

    _mols : list, shape(num_valid_mol_objects), initialise as `[]`
        List of mol objects which can be calculated on to provide molecular fingerprints.

    _fingerprints : list[list[str]], shape(num_fingerprints, num_fingerprint_chars)
        List of obtained fingerprint vectors where mol objects have been operated on.

    _fingerprint_representations : list, shape(num_fingerprints)
        List of representations which are succesfull in eventual generation of fingerprints.
        Used by `FingerprintFactory` in `produce_dict`, `produce`_json` and `produce_dataframe` methods.

    nbits : int
        Number of characters in the molecular fingerprint / elements in the bit vector.
        Used for naming the columns output by `produce_dataframe`.
        Not a class attribute since it is prefferable to specify a value at instantiation for some fingerprints
        like ECFP and FCFP, whereas other fingerprints may use a hardcoded value.
        
    Methods
    -------
    representation_converter(representation) --> converts molecule representation (smiles etc) into a mol object.
    representation_to_mol(self, *compound_representation) --> Convert representation to rdkit mol (default SMILES)
    mol_to_fingerprint(self, mol) --> NotImplemented (To be subclassed)
    obtain_fingerprints(self) --> Public method for factory to produce fingerprints
    produce_list(self, as_strings=True) --> return fingerprints as list
    produce_dict(self) --> return fingerprints as dict
    produce_json(self) --> return fingerprints as json string
    produce_array(self) --> return fingerprints as numpy array
    produce_dataframe(self) --> return fingerprints as pandas DataFrame

    __str__ --> Outputs Factory status.
    __len__ --> Length of `self._mols`.
    __repr__ --> Outputs public arguments.

    Notes
    -----
    Two list attributes exist for tracking molecule representations within the FingerprintFactory:
     * _representations
     * _fingerprint_representations
     The former is used to store the passed representations which are succesfully converted into mol objects.
     The latter is used to store the representations which are converted to mol objects which are then succesfully
     opperated on to produce a molecular fingerprint.

    These separate lists are maintained by FingerprintFactory so that when reporting products of the factory either
    by `produce_dict`, `produce_json`, or `produce_dataframe`, a succesful 1:1 match can be reported.
    Alternatively, if a representation could generate a mol object which was not capable of being operated on to
    obtain a fingerprint, the output would be mismatched and therefore provide incorrect products to the user.
    """

    def __init__(self):
        """Initialise the FingerprintFactory object.
        """
        self._representations = []
        self._mols = []
        self._fingerprints = []
        self._fingerprint_representations = []

    @staticmethod
    def representation_converter(representation):
        """Convert molecular representation / file into a mol object capable of being calculated over.
        Must accept single representation as input and return a singular correpsonding mol like object.
        Default converts from SMILES string to `rdkit.Chem.rdchem.Mol` object.

        Notes
        -----
        This is provided with common conversion functionality however users can readily subclass to create their
        own converters i.e. for InChI conversion or reading from a structural file etc.

        Parameters
        ----------
        representation : str
            Smiles string representation of olecule to be converted to mol object.

        Returns
        -------
        mol : `rdkit.Chem.rdchem.Mol`
            Singular mol object as implemented by `rdkit`.
        """
        mol = Chem.MolFromSmiles(representation)
        return mol

    def mol_to_fingerprint(self, mol):
        """Calculates and returns the fingerprint of a singular mol object as a numpy array.

        Notes
        -----
        This function is intended to be overloaded to calculate different fingerprints which must obey the form:

            mol_to_fingerprint(mol) --> fingerprint (np.array)

        The returned numpy array **must** be a row vector if the fingerprint is 1D where each element is a specific bit.
        If the fingerprint is multi-dimensional then the `n`D vector can be returned without constraint.
        """
        return NotImplemented

    def obtain_representations(self, compound_representation):
        """Converts representation of compound(s) into molecule objects.
        Default implementation converts from SMILES string to`rdkit.Chem.rdchem.Mol`.

        Different conversions can be performed by subclassing and overleading `representation_converter`
        with form:

            func(representation) --> mol.

        Parameters
        ----------
        *compound_representation : iterable, shape(num_entries)
            List of or singular entry of a compound representation i.e. SMILES, InCHi etc...

        Returns
        -------
        None
            Updates `self._representations`, `self._mols`.
        """
        valid_mols, valid_compunds = [], []
        for compound in compound_representation:
            try:
                mol = self.representation_converter(compound)
                valid_mols.append(mol)
                valid_compunds.append(compound)
            except:
                warn(F'Unable to convert {compound}')
                # bare exception used as user defined converters could invoke errors not previously considered.

        self._representations, self._mols = valid_compunds, valid_mols

    def obtain_fingerprints(self):
        """User method to obtain the fingerprints for all previously generated mol objects.
        Each fingerprint is obtained iteratively via the `mol_to_fingerprint` which is implementation specific.
        Generated fingerprints must be a numpy array.

        Raises
        ------
        AssertionError
            Raised if produced fingerprint is not a numpy array.

        Returns
        -------
        None
            Updates `self._fingerprints` and `self._fingerprint_representations`.
        """
        for rep, mol in zip(self._representations, self._mols):
            try:
                fp = self.mol_to_fingerprint(mol)
                assert isinstance(fp, np.ndarray), 'Fingerprint must be generated as a 1D or `n`D numpy array.'
                self._fingerprints.append(fp)
                self._fingerprint_representations.append(rep)
            except:
                warn(F'Error encountered obtaining fingerprint of {rep}')

    def produce_list(self, as_bit_string=False):
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
        if not as_bit_string:
            product = self._fingerprints
        else:
            try:
                product = [''.join(fp.astype(str)) for fp in self._fingerprints]
                # fingerprints expected to be numpy arrays, hence should support `.astype` conversion.
            except TypeError:
                raise BitVectorRepresentationError('Unable to represent fingerprint as bit string.')
        return product

    def produce_dict(self, as_bit_string=False):
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
        fingerprints = self.produce_list(as_bit_string=as_bit_string)
        product = {str(k): v for k, v in zip(self._fingerprint_representations, fingerprints)}
        return product

    def produce_json(self, as_bit_string=False):
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
        fp_dict = self.produce_dict(as_bit_string=as_bit_string)
        if not as_bit_string:
            to_serialise = {k: fp_dict[k].tolist() for k in fp_dict}  # convert np arrays to list for serialisation
        else:
            to_serialise = fp_dict
        try:
            product = json.dumps(to_serialise)
        except TypeError:
            raise JsonSerialisationError('Unable to serialise representation and fingerprint data to JSON format.')
        return product

    def produce_array(self, as_bit_string=False, as_type=str):
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
        fingerprints = self.produce_list(as_bit_string=as_bit_string)
        product = np.array(fingerprints)
        if not as_bit_string:
            try:
                product = product.astype(as_type)
            except ValueError:
                raise FingerprintRepresentationError('Unable to convert fingerprint to specified type. '
                                                     'Ensure all fingerprints are equal length for all molecules.')
        return product

    def produce_series(self, as_bit_string=False, as_type=str):
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
        data = list(self.produce_array(as_bit_string=as_bit_string, as_type=as_type))
        representations = [str(rep) for rep in self._fingerprint_representations]
        product = pd.Series(data=data, index=representations)
        return product

    def produce_dataframe(self, as_type=str):
        """Returns a pandas DataFrame of molecule fingerprints, indexed by the representations.
        Column headers are integer indices of each character / bit in the fingerprint, indexed from 0.
        Dataframes of fingerprints can only be returned when fingerprints obtained are each 1D vectors.
        If fingerprints are expected to be 2D or greater, consider calling `produce_series` or `produce_array`
        instead.

        >>> FingerprintFactory.produce_dataframe(as_type=int)
                        0   1   2   3   4
        'smiles_0'      1   1   0   0   1
        ...             ...
        'smiles_n'      1   0   0   0   1
        [n rows x 5 columns]

        Parameters
        ----------
        as_type : numpy array type {str, int, float} (default = str)
            Data type which each numpy array element in the series should be returned as.

        Raises
        ------
        BitVectorRepresentationError
            If a fingerprint canot be converted into a bit string / vector.

        FingerprintRepresentationError
            If unable to convert the output array to the desired type.
            Typically would occur when all obtained fingerprints are not the same length.
            Can also be raised if fingerprints are unable to be represented in 2D pandas DataFrame.

        Returns
        -------
        product : pd.DataFrame, shape(num_fingerprints, fingerprint_length)
            Pandas DataFrame with each row as a singular fingerprint and each column as a singular bit / element
            of the bit vector.
            The datarame is indexed with the representation of each molecule.
        """
        representations = [str(rep) for rep in self._fingerprint_representations]
        data = self.produce_array(as_type=as_type)
        try:
            product = pd.DataFrame(data=data, index=representations)
        except ValueError:
            raise FingerprintRepresentationError('ingerprints must be 1D vectors to convert to dataframe. '
                                                 'Call `produce_series` if fingerprints are `n` dimensional vectors.')
        return product

    def get_mols(self):
        """Getter method to access list of generated mol objects from the factory.

        Returns
        -------
        get : list, shape(num_mol_objects)
            List of mol objects generated when calling `representation_to_mol`.
        """
        get = self._mols
        return get

    def get_representations(self, validity='mol'):
        """Getter method to access list of representations.
        Since two lists of representations are maintained (converted to mols, and converted to fingerprints)
        user must give a keyword specifiying which representation list to recieive.

        Notes
        -----
        A list of user input representations when calling `representation_to_mol` is not maintained and so
        cannot be retrieved.

        Parameters
        ----------
        validity : str {'mol', 'fingerprint'}
            Keyword to specifiy which representation list shoud be returned to the user.
            * 'mol' --> `self._representations`
            * `fingerprint` --> `self._fingerprint_representation`

        Returns
        -------
        get : list, shape(num_representations)
            List of representations.
        """
        rep_dict = {
            'mol': self._representations,
            'fingerprint': self._fingerprint_representations
        }
        assert validity.lower() in rep_dict, F'Specified Representation must be in {list(rep_dict.keys())}'
        get = rep_dict[validity]
        return get

    def get_nbits(self):
        """Getter method to access the number of characters / bits expected in the fingerprint.

        Returns
        -------
        get : int
            Number of characters / bits in the fingerprint.
        """
        get = self.nbits
        return get

    def __str__(self):
        """Details number of recieved representations, number of mols, and number of fingerprints calculated.

        Returns
        -------
        out : str
            String representation of FingerprintFactory.
        """
        out = F'Factory Record: ' \
              F'{len(self._representations)} representations recieved, ' \
              F'{len(self._mols)} converted to mols, ' \
              F'and {len(self._fingerprints)} fingerprints calculated.'
        return out

    def __len__(self):
        """Returns number of fingerprints which have been calculated.

        Returns
        -------
        length : int
            Number of fingerprints which FingerprintFactory has generated.
        """
        length = len(self._fingerprints)
        return length

    def __repr__(self):
        """Returns string representation of public class attributes sourced from `vars(self)`.

        Returns
        -------
        public_args : str
            String representation of dict containing public class attributes.
        """
        object_vars = vars(self)
        public_args = json.dumps({k: object_vars[k] for k in object_vars if k[0] != '_'})
        return public_args
