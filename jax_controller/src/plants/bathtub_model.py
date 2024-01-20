from plants.base_plant import BasePlant
from math import sqrt


class BathtubModel(BasePlant):

    def __init__(self, A, C, target) -> None:
        self.A = A
        self.C = C
        self.H = target
        self.V = sqrt(2 * 9.81 * self.H)
        self.Q = self.V * self.C
        self.target = target
    
    def get_state(self):
        pass

    def save_init_state(self):
        """
        TODO save the initial state of the model to a dictionary and call this inside constructor
        """
        pass

    def reset_plant(self):
        # self.H = self.target
        # # TODO trengs disse to her?
        # self.V = sqrt(2 * 9.81 * self.H)
        # self.Q = self.V * self.C
        pass

    def compute_plant_output(self, controller_params, disturbance, Q, A, error_history, incoming_error, compute_control_signal: callable):
        # TODO implement controller params so that we can differentiate on params
        control_signal = compute_control_signal(controller_params, error_history, incoming_error)
        volume_change = control_signal + disturbance - Q
        water_height_change = volume_change / A
        # TODO check if this should be + or -
        self.H += water_height_change
        return self.H
