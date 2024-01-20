from abc import ABC, abstractmethod


class BaseController(ABC):

    @abstractmethod
    def init_params(self):
        """
        Initialize the controller parameters.
        """
        pass
    
    # def init_state_history(self):
    #     """
    #     """
    #     self.state_history = []

    # # TODO try to only use error history
    # def save_state(self, timestep, incoming_error, disturbance, plant_output, outgoing_error, error_history):
    #     """
    #     Save the state of the system.
    #     """
    #     self.state_history.append({
    #         'timestep': timestep,
    #         'incoming_error': incoming_error,
    #         'disturbance': disturbance,
    #         'plant_output': plant_output,
    #         'outgoing_error': outgoing_error,
    #         'error_history': error_history
    #     })
