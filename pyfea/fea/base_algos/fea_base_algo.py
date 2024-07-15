import abc


class FeaBaseAlgo(metaclass=abc.ABCMeta):
    """
    All base algorithms plugged into an FEA should inherit this class,
    as our implementation of FEA calls all these methods.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Asserts that a given class is only a subclass if they implement all its methods.
        @param subclass: the subclass to check for these methods.
        """
        return (
            hasattr(subclass, "base_reset")
            and callable(subclass.base_reset)
            and hasattr(subclass, "get_solution_at_index")
            and callable(subclass.get_solution_at_index)
            and hasattr(subclass, "update_worst")
            and callable(subclass.update_worst)
            and hasattr(subclass, "from_kwargs")
            and callable(subclass.from_kwargs)
            and hasattr(subclass, "run")
            and callable(subclass.run)
            and hasattr(subclass, "update_bests")
            and callable(subclass.update_bests)
        )

    @classmethod
    @abc.abstractmethod
    def _from_kwargs(cls, function, domain, params):
        """
        The method used to construct parameter input for FEA base algorithms.
        @param function: the objective function that the base algorithm minimizes.
        This will be passed as an object of type Function, as defined in function.py.
        @param domain: the space that the base algorithm minimizes over, as a numpy array
        of size (dim,2). This will take into account the factor structure being used.
        @param params: a dictionary of keyword arguments used as parameters to construct
        the base algorithm.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _base_reset(self):
        """
        Reset any values of the base algorithm that need it here. For instance,
        reinitialize particle velocities if they've been shrinking.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_solution_at_index(self, idx):
        """
        Find the algorithm's current solution for a given variable of our function for
        use in the compete step of FEA.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_worst(self, context):
        """
        Implement the second half of the FEA algorithm's share step for population-based algorithms
        here. Or don't. This is one of the least important parts of the FEA algorithm.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        Run the base algorithm through an appropriate number of iterations.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_soln(self):
        """
        Run the base algorithm through an appropriate number of iterations.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_bests(self):
        """
        Update the base algorithm's evaluation of the fitness function according to new information
        about the global context vector in FEA's share step.
        """
        raise NotImplementedError
