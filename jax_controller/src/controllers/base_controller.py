from abc import ABC, abstractmethod


class BaseController(ABC):

    # TODO find out what this class should be

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass
