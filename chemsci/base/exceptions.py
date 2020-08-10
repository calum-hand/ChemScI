class StandardRepresentationError(Exception):
    """Named Exception raised when invalid representation selection passed to `StandardFeatureTransformer` constructor
    method.
    """

class BitVectorRepresentationError(Exception):
    """Named Exception raised when unable to convert the numpy array representation of molecular fingerprint
    to a string.
    """
    pass


class FingerprintRepresentationError(Exception):
    """Named Exception raised when fingerprints are unable to be interacted with due to incorrect types or mismatched
    numpy array sizes.
    """
    pass


class JsonSerialisationError(Exception):
    """Named Exception raised if FingerprintFactory is unable to correctly serialise data into JSON format.
    """
    pass
