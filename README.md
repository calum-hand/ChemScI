# ChemSci

## Goals
* Provide a consistent interface for the featurisation of molecules or other chemical systems for informatics studies
* Allow for these featurisations to be easily saved to a wide variety of formats
* Allow for novel featurisations to easily be incorporated / implemented from literature within minimal boiler plate required from contributing authors

ChemSci is designed to facilitate an entirely modular approach to chemical featurisation, where the featurisation of a chemical system does not assume it originates from a SMILES string etc.
This prevents commonly used tools from being subtly hard-coded into the ChemSci and ensures a flexible API is maintained.

Through providing a consistent API, it is hoped that newly developed featurisations for cheminformatics can be readily incorporated and hence provide easy access to the informatics community.

## TODO
### Factory object
1. Inherit from sklearn transformer mixin with appropriate methods
2. representation and featurisation converters are passed as composition to main factory or callable objects
3. Add more output file formats to the factory / roller whatever (i.e. YAML / SQLITE)
### Features
1. Try and add a literature example as a demo for novel incorporation (if code + suitable license)
2. Use pubchem to really demonstrate non hard coded example
### Development
1. Get new conda env set up for the project
2. Doccumentation!!!
2. Get environment.yml set up rather than requirements.txt for full conda experience
3. Get some tests written
4. Get some CI up and running (somewhere free...)
5. Get a docker file in there for good measure! 