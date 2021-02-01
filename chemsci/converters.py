import time

from pubchempy import Compound

# ----------------------------------------------------------------------------------------------------------------------


def no_conversion_required(representation):
    """Used when no conversion from the current representation is required.
    This is useful when dealing with libraries which create the featurisation directly from the representation and
    hence do not require conversion to another form.

    Parameters
    ----------
    representation

    Returns
    -------

    """
    return representation

# ----------------------------------------------------------------------------------------------------------------------


class pubchem_conv:
    """

    """

    def __init__(self, crawl_delay=2.0):
        """

        Parameters
        ----------
        crawl_delay
        """
        self.crawl_delay = abs(float(crawl_delay))

    def __call__(self, representation):
        """

        Parameters
        ----------
        representation

        Returns
        -------

        """
        compound = Compound.from_cid(representation)  # calls PubChem API
        time.sleep(self.crawl_delay)
        return compound

# ----------------------------------------------------------------------------------------------------------------------
