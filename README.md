# ChemSci
![Build Status](https://github.com/Bundaberg-Joey/ChemScI/workflows/ChemScI/badge.svg)

## Goals
* Provide a consistent interface for the featurisation of molecules or other chemical systems for informatics studies
* Allow for these featurisations to be easily saved to a wide variety of formats
* Allow for novel featurisations to easily be incorporated / implemented from literature within minimal boiler plate required from contributing authors

ChemSci is designed to facilitate an entirely modular approach to chemical featurisation, where the featurisation of a chemical system does not assume it originates from a SMILES string etc.
This prevents commonly used tools from being subtly hard-coded into the ChemSci and ensures a flexible API is maintained.

Through providing a consistent API, it is hoped that newly developed featurisations for cheminformatics can be readily incorporated and hence provide easy access to the informatics community.
**In development of new molecule featurisations, users are encourgaed to submit their own featurisations to `ChemSci` so that a consistent api for all featurisations is maintained.**

## FeatureFactory
The core object of ChemSci is the `FeatureFactory` which can be used as a `sklearn` transformer or as a stand alone object in informatics studies.
Within the `FeatureFactory` are the `converter` and `featuriser` callables which are specified at initialisation, either through default keywords or custom callables.
The `converter` callable takes a standard representation of a chemical system and converts it into a workable obeject from which featurisations can be derived.
The `featuriser` callable takes the workable object and generates the featurisation from it.

```python
from chemsci.factory import FeatureFactory

ff = FeatureFactory(converter='smiles', featuriser='maccs')  
# create featuriser which reads smiles strings and calculates maccs fingerprints

X = ['smiles_1', 'smiles_2', 'smiles_n']
out = ff.fit_transform(X)
```

### Default converters and featurisers
Default `converter`s:
* smiles
* smarts
* inchi
* mol (path to file)
* pdb (path to file)
* pubchem (CID number)
* none (when no conversion is required)

Default `featuriser`s:
* maccs (Molecular Access fingerprint)
* avalon (Avalon fingerprint)
* daylight (daylight fingerprint)
* ecfp_4_1024 (1024 bit ECFP of radius 4)
* ecfp_6_1024 (1024 bit ECFP of radius 6)
* ecfp_4_2048 (2048 bit ECFP of radius 4)
* ecfp_6_2048 (2048 bit ECFP of radius 6)
* fcfp_4_1024 (1024 bit ECFP of radius 4)
* fcfp_6_1024 (1024 bit ECFP of radius 6)
* fcfp_4_2048 (2048 bit ECFP of radius 4)
* fcfp_6_2048 (2048 bit ECFP of radius 6)


### Custom Converters and Featurisers:
Any callable can be passed in place of a `converter` and `featuriser`, with several customs available in `custom_featurisers` and `converters`.
For example, both the "MAP4" molecular fingerprints (DOI 10.1186/1758-2946-5-26) and pubchempy fingerprints are available to be used.

```python
from chemsci.factory import FeatureFactory
from chemsci.custom_featurisers import Map4Fingerprint

ff = FeatureFactory(converter='smiles', featuriser=Map4Fingerprint())
```

The contracts of the callables are as below where any type can be used as input but the output of `featuriser` must be a numpy array.

```python
def converter(representation):
    """
    Parameters
    ----------
    representation: any
        any type input.
    
    Returns
    -------
    out : any
    """
    out = custom_conversion(representation)
    out


def featuriser(mol):
    """
    Parameters
    ----------
    representation: any
        any type input.
    
    Returns
    -------
    out : np.ndarray
    """
    out = custom_featuisation(representation)
    out

```

### Integration with `sklearn`
The `FeatureFactory` object inherits from `sklearn.base.TransformerMixin` and so can readily be included in any pipeline / workflow with ease.

### Convenience Outputs
As with pandas dataframes, the results of a `FeatureFactory` transformation can be returned as given datatypes / saved to given file structures with ease due to the provided convenience functions:
* `to_list()`
* `to_dict(convert_string=False)`
* `to_arrary(as_type=None)`
* `to_series(index=None, name=None, as_type=None)`
* `to_csv(self, path, index=None, name=None, as_type=None, sep=',', save_index=True, columns=None, header=True)`
* `to_json(self, path=None)`
* `to_yaml(self, path)`

## Installation
```
git clone https://github.com/Bundaberg-Joey/ChemScI.git
conda create --name <env_name> --file=environment.yml
pip install .
```