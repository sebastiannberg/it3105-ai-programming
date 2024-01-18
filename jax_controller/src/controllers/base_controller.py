from abc import ABC, abstractmethod


class BaseController(ABC):

    def __init__(self) -> None:
        self.control_signal = 0

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass
    
    @abstractmethod
    def update_control_signal(self, error):
        """
        Update the control signal based on given error.
        """
        pass

    def init_error_history(self):
        """
        Initialize the error history.
        """
        # TODO this may not be doable because of integrating and mse over error
        self.error_history = [0]

    def save_error(self, error):
        """
        Save the error in error history.
        """
        self.error_history.append(error)
