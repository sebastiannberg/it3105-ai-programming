from plants.base_plant import BasePlant

import jax.numpy as jnp


class BathtubModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        # TODO check for valid dict, does it contain the necessary keys? throw error if not
        self.init_plant_state = init_plant_state
    
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
        return state
