from abc import ABC, abstractmethod


class BasePlant(ABC):

    @abstractmethod
    def reset_plant(self):
        """
        Reset plant to initial state.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Updates the plant. Returns the output.
        """
        pass
