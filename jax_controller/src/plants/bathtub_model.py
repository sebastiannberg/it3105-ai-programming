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

    def reset_plant(self):
        # self.H = self.target
        # # TODO trengs disse to her?
        # self.V = sqrt(2 * 9.81 * self.H)
        # self.Q = self.V * self.C
        pass

    def compute_plant_output(self, controller_params, disturbance, compute_control_signal):
        control_signal = compute_control_signal(controller_params, self.target - self.H)
        volume_change = control_signal + disturbance - self.Q
        water_height_change = volume_change / self.A
        # TODO check if this should be + or -
        self.H += water_height_change
        return self.H
