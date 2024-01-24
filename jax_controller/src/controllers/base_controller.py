from abc import ABC, abstractmethod


class BaseController(ABC):

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass

    @abstractmethod
    def update_controller(self, params, state):
        """
        Update the controller given some parameters and a state dictionary.
        The control signal is within the state dictionary.
        """
        pass
