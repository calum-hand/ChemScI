class BitVectorRepresentationError(Exception):
    """Named Exception raised when unable to convert the numpy array representation of molecular fingerprint
    to a string.
    """
    pass


class JsonSerialisationError(Exception):
    """Named Exception raised if FingerprintFactory is unable to correctly serialise data into JSON format.
    """
    pass
