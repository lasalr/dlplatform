from DLplatform.aggregating import Aggregator
from DLplatform.parameters import Parameters

class RadonPoint(Aggregator):
    """
    Provides a method to aggregate n models using the Radon point method
    """

    def __init__(self, name="Radon point"):
        '''

        Returns
        -------
        None
        '''
        Aggregator.__init__(self, name=name)

    def calc_radon_point(self):
        pass
        #TODO Need to write this
        return self