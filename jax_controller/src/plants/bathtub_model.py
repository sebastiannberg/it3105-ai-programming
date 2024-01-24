from plants.base_plant import BasePlant

import jax.numpy as jnp
import jax


class BathtubModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        if self.check_valid_init_state(init_plant_state):
            self.init_plant_state = init_plant_state
    
    def check_valid_init_state(self, init_plant_state):
        required_keys = ("A", "C", "H", "V", "Q", "target")
        if not all(key in init_plant_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in init_plant_state]
            raise ValueError(f"Missing keys in init_plant_state: {missing_keys}")
        return True

    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        # Specify float type for control_signal because of jax trace
        volume_change = jnp.array(state["control_signal"], dtype=jnp.float32) + disturbance - state["Q"]
        water_height_change = volume_change / state["A"]
        state["H"] += water_height_change
        # Use jnp.sqrt because of jax trace
        state["V"] = jnp.sqrt(2 * 9.81 * state["H"])
        state["Q"] = state["V"] * state["C"]
        state["plant_output"] = state["H"]
        state["current_error"] = state["target"] - state["plant_output"]
        # jax.debug.print("current error {x}", x=state["current_error"])
        return state
