from abc import ABC, abstractmethod


class BaseController(ABC):

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass

    @abstractmethod
    def update_params(self, loss_gradient):
        """
        Update the controller parameters based on the gradient.
        """
        pass

    def init_error_history(self):
        """
        """
        self.error_history = []

    def save_error(self, error):
        self.error_history.append(error)

    def init_plant_history(self):
        """
        Initialize the plant history.
        """
        # TODO maybe jnp array for calculating gradients
        self.plant_history = []

    def save_plant_output(self, compute_plant_output: callable):
        """
        Save the plant output in the plant history.
        """
        # TODO Seems like it is not saved as a callable when trying to call func in mse function
        self.plant_history.append(compute_plant_output)
        return compute_plant_output
