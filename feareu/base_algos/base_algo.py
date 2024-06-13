import abc

class BaseAlgo(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'reset') and callable(subclass.reset) and hasattr(subclass, 'get_solution_at_index') and callable(subclass.get_solution_at_index) and hasattr(subclass, 'update_worst') and callable(subclass.update_worst) and hasattr(subclass, from_kwargs) and callable(subclass.from_kwargs))

    @classmethod
    @abc.abstractmethod
    def from_kwargs(cls, function, domain, kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_solution_at_index(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def update_worst(self):
        raise NotImplementedError
