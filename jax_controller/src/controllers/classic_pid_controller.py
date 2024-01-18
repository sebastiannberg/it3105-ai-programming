from controllers.base_controller import BaseController


class ClassicPIDController(BaseController):

    def __init__(self) -> None:
        super().__init__()
        
    def init_params(self):
        self.kp = 0.1
        self.ki = 0.01
        self.kd = 0.001

    def update_control_signal(self, error):
        self.control_signal = self.kp * error + self.ki * sum(self.error_history) + self.kd * (error - self.error_history[-1])
