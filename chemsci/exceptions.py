class BitVectorRepresentationError(Exception):
    """Unable to convert the numpy array representation of molecular fingerprint to a string.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class FingerprintRepresentationError(Exception):
    """Features are unable to be interacted with due to incorrect types or mismatched numpy array sizes.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class JsonSerialisationError(Exception):
    """FeatureFactory is unable to correctly serialise data into JSON format.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class UserSelectionError(Exception):
    """Invalid selection is made by the user, typically when initialising objects with default args.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class ConversionError(Exception):
    """Unable to convert the passed representation into a workable object for featurisation.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class FeaturisationError(Exception):
    """Unable to featureise the passed converted representation."""
    pass

# ----------------------------------------------------------------------------------------------------------------------
