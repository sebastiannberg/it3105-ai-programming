from controllers.base_controller import BaseController


class ClassicPIDController(BaseController):
        
    def init_params(self):
        self.kp = 0.1
        self.ki = 0.01
        self.kd = 0.001

    def init_error_history(self):
        self.error_history = []
