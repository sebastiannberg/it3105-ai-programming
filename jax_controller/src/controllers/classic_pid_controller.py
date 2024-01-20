from controllers.base_controller import BaseController
import numpy as np
import jax.numpy as jnp


class ClassicPIDController(BaseController):
        
    def init_params(self):
        # kp, ki and kd
        return np.array([0.1, 0.01, 0.001])

    def compute_control_signal(self, params, state):
        if not error_history:
            # First iteration of the controller, control signal is zero
            return 0.0
        return jnp.dot(controller_params, jnp.array([incoming_error, jnp.sum(jnp.array(error_history)), incoming_error - error_history[-1]]))
