from controllers.base_controller import BaseController
import numpy as np
import jax.numpy as jnp


class ClassicPIDController(BaseController):

    def __init__(self, kp, ki, kd) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def init_params(self):
        # kp, ki and kd
        return jnp.array([self.kp, self.ki, self.kd])
    
    def update_controller(self, params, state):
        if not state["error_history"]:
            # Edge case because error_history has no values
            state["control_signal"] = jnp.dot(params, jnp.array([state["current_error"], 0, state["current_error"]]))
        else:
            state["control_signal"] = jnp.dot(params, jnp.array([state["current_error"], jnp.sum(jnp.array(state["error_history"])), state["current_error"] - state["error_history"][-1]]))
        return state
