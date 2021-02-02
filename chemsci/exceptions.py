class FeatureRepresentationError(Exception):
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


class ConversionWarning(UserWarning):
    """Unable to convert the passed representation into a workable object for featurisation.
    """
    pass

# ----------------------------------------------------------------------------------------------------------------------


class FeaturisationWarning(UserWarning):
    """Unable to featureise the passed converted representation."""
    pass

# ----------------------------------------------------------------------------------------------------------------------
