from warnings import warn
import json

import pandas as pd
import numpy as np

from rdkit import Chem

# TODO : Need to sort out if the `mol_to_fingerprint` should produce strings OR numbers (makes more sense for `produce` fnctons to output numbers than strings but conversion gets iffy
# TODO : Once set up and running for basic fingerprints, need to see about how to get running for matrix fingerprints etc

class FingerprintFactory:
    """Parent class used to calculate various molecular fingerprints and return them to the user in a variety of ways.
    Expects molecular fingerprints to be equal length regardless of molecule.

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
        """Calculates and returns the fingerprint of a singular mol object.

        Notes
        -----
        This function is intended to be overloaded to calculate different fingerprints which must obey the form:

            mol_to_fingerprint(mol) --> fingerprint (np.array)

        The returned object must have each bit / vector element represented as a single element in a numpy array.
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

        self._representations, self._mols = valid_compunds, valid_mols

    def obtain_fingerprints(self):
        """User function to calculate the fingerprints for all mol objects.
        Calculation occurs iteratively for each mol via the `mol_to_fingerprint` method.

        Returns
        -------
        None
            Updates `self._fingerprints` and `self._fingerprint_representations`.
        """
        for rep, mol in zip(self._representations, self._mols):
            try:
                fp = self.mol_to_fingerprint(mol)
                self._fingerprints.append(fp)
                self._fingerprint_representations.append(rep)
            except:
                warn(F'Error encountered obtaining fingerprint of {rep}')

    def produce_list(self, as_strings=True):
        """Returns a list of molecule fingerprints.
        User can specify if the list is of fingerprint strings or a list of lists where each element
        of the nested list is a single fingerprint character i.e.:

        >>> FingerprintFactory.produce_list()
        ['11001', '10001',..., '00001']

        >>> FingerprintFactory.produce_list(as_strings=False)
        [['1', '1', '0', '0', '1'], ['1','0', '0', '0', '1'], ..., ['0', '0', '0', '0', '1']

        Returns
        -------
        product : list[str] or list[list[str]], shape(num_fingerprints, )
            List of fingerprints, either represented as strings or list of individiual elements.
        """
        if as_strings:
            product = [''.join(fp) for fp in self._fingerprints]
        else:
            product = self._fingerprints
        return product

    def produce_dict(self):
        """Returns a dict of molecule representations and corresponding fingerprints.

        Returns
        -------
        product : dict, {representation: fingerprint}, {str: list[str]}
            Dictionary of molecular representations and corresponding fingerprint.
        """
        product = {k: v for k, v in zip(self._fingerprint_representations, self._fingerprints)}
        return product

    def produce_json(self):
        """Returns a JSON string of molecule representations and corresponding fingerprints.
        Equivalent to running:

        >>> product = FingerprintFactory.produce_dict()
        >>> json_product = json.dumps(product)

        Returns
        -------
        product : dict, {representation: fingerprint}, {str: list[str]}
            Dictionary of molecular representations and corresponding fingerprint.
        """
        fp_dict = self.produce_dict()
        product = json.dumps(fp_dict)
        return product

    def produce_array(self, as_type=str):
        """Returns a numpy array of molecule fingerprints of type `str`.
        Equivalent to running:

        >>> product = FingerprintFactory.produce_list(as_strings=False)
        >>> product_arr = np.array(product).astype(str)

        Parameters
        ----------
        as_type : numpy array type {str, int, float} (default = str)
            Data type which the contents of each numpy array should be returned as.

        Returns
        -------
        product : np.array, shape(num_fingerprints, fingerprint_length)
            Numpy array of fingerprints where each row is a fingerprint and each column is a specific bit.
        """
        product = np.array(self._fingerprints).astype(as_type)
        return product

    def produce_series(self, name='Fingerprint'):
        """Returns a pandas Series of molecular fingerprints where each entry is the entire fingerprint
         represented as a string.

        Parameters
        ----------
        name : str (default = 'Fingerprint')
            Name to be given to the pandas Series object.

        Returns
        -------
        product : pd.Series, shape(num_fingerprints)
            Pandas Series object containing string representations of calculated fingerprints.
            Indexed by representations.
        """
        data = self.produce_list(as_strings=True)
        product = pd.Series(name=name, data=data, index=self._fingerprint_representations)
        return product

    def produce_dataframe(self):
        """Returns a pandas DataFrame of molecule fingerprints, indexed by the representations.
        Column headers are integer indices of each character / bit in the fingerprint, indexed from 0.

        Returns
        -------
        product : pd.DataFrame, shape(num_fingerprints, fingerprint_length)
            Pandas DataFrame with each row as a singular molecule, indexed with the representation.
        """
        columns = [i for i in range(self.nbits)]
        data = self.produce_list(as_strings=False)
        product = pd.DataFrame(data=data, columns=columns, index=self._fingerprint_representations)
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