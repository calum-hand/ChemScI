import time

from pubchempy import Compound

# ----------------------------------------------------------------------------------------------------------------------


def no_conversion_required(representation):
    """Used when no conversion from the current representation is required, simply returns the passed representation.
    This is useful when dealing with libraries which create the featurisation directly from the representation and
    hence do not require conversion to another form.

    Parameters
    ----------
    representation : Any
        String or other representation of a molecular structure / system.

    Returns
    -------
    representation : Any
        String or other representation of a molecular structure / system.
    """
    return representation

# ----------------------------------------------------------------------------------------------------------------------


class pubchem_conv:
    """Create molelcular representation from [PubChem API](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest-tutorial).
    Uses `pubchempy` as a backend.
    """

    def __init__(self, crawl_delay=2.0):
        """
        Parameters
        ----------
        crawl_delay : float
            (default = 2.0)
            Good practise is to use a crawl delay when accessing the PubChem API.
            Passed value must be a non negative float.
        """
        self.crawl_delay = abs(float(crawl_delay))

    def __call__(self, representation):
        """Create `pubchempy.Compound` object from passed CID representation.

        Parameters
        ----------
        representation : int
            CID integer as per `pubchempy` doccumentation for `from_cid`.

        Returns
        -------
        compound : pubchempy.Compound
            Pubchempy represenation of molecule, accessed from the PubChem database.
        """
        compound = Compound.from_cid(representation)  # calls PubChem API
        time.sleep(self.crawl_delay)  # ensure good scraping practise
        return compound

# ----------------------------------------------------------------------------------------------------------------------
