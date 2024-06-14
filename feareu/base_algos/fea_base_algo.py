import abc

class FeaBaseAlgo(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'base_reset') 
                and callable(subclass.base_reset) 
                and hasattr(subclass, 'get_solution_at_index') 
                and callable(subclass.get_solution_at_index) 
                and hasattr(subclass, 'update_worst') 
                and callable(subclass.update_worst) 
                and hasattr(subclass, 'from_kwargs') 
                and callable(subclass.from_kwargs)
                #and hasattr(subclass, 'run')
                #and callable(subclass.run)
                #and hasattr(subclass, 'update_bests')
                #and callable(subclass.update_bests)
                )

    @classmethod
    @abc.abstractmethod
    def from_kwargs(cls, function, domain, params):
        raise NotImplementedError

    @abc.abstractmethod
    def base_reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_solution_at_index(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def update_worst(self, context):
        raise NotImplementedError

    #@abc.abstractmethod
    #def run(self):
    #    raise NotImplementedError

    #@abc.abstractmethod
    #def update_bests(self):
    #    raise NotImplementedError
