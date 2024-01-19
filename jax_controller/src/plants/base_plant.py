from abc import ABC, abstractmethod


class BasePlant(ABC):

    def __init__(self) -> None:
        super().__init__()

    # @abstractmethod
    # def reset_plant(self):
    #     """
    #     Reset plant to initial state.
    #     """
    #     pass