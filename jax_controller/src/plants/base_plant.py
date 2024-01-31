from abc import ABC, abstractmethod


class BasePlant(ABC):

    @abstractmethod
    def check_valid_init_state(self, init_plant_state: dict):
        """
        Returns True if the dictionary contains the necessary keys for this plant
        to work properly.
        """
        pass

    @abstractmethod
    def get_init_plant_state(self):
        """
        Returns the dictionary containing the initial plant state variables.
        """
        pass

    @abstractmethod
    def update_plant(self, state: dict, disturbance):
        """
        Updates the plant based on the state dictionary. The values in state is modified
        and returned when done.
        """
        pass

