from abc import ABC, abstractmethod


class BasePlant(ABC):

    # TODO kanskje ikke abstract?
    @abstractmethod
    def reset_plant(self):
        """
        Reset plant to initial state.
        """
        pass

    @abstractmethod
    def update(self, control_signal, disturbance):
        """
        Updates the plant. Returns the error.
        """
        pass
