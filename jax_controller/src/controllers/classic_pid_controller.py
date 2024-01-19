from controllers.base_controller import BaseController
import numpy as np


class ClassicPIDController(BaseController):

    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate
        
    def init_params(self):
        # kp, ki and kd
        self.params = np.array([0.1, 0.01, 0.001])

    def compute_control_signal(self, controller_params, error):
        if not self.plant_history:
            # First iteration of the controller, control signal is zero
            return 0
        return np.dot(controller_params, [error, sum(self.error_history), (error - self.error_history[-1])])

    def update_params(self, loss_gradient: callable, controller_params, disturbance, compute_control_signal, plant_history):
        self.params -= self.learning_rate * loss_gradient(controller_params, disturbance, compute_control_signal, plant_history)
