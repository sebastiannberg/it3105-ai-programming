from abc import ABC, abstractmethod


class BaseController(ABC):

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass

    @abstractmethod
    def init_error_history(self):
        """
        Initialize the error history.
        """
        pass
