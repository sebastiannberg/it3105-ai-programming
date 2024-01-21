from plants.base_plant import BasePlant

import jax.numpy as jnp


class CournotModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        self.init_plant_state = init_plant_state
    
    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        # TODO enforce q1 and q2 0<q<1? do it another place?
        state["q1"] += state["control_signal"]
        state["q2"] += disturbance
        state["q"] = state["q1"] + state["q2"]
        state["price"] = state["max_price"] - state["q"]
        state["producer_one_profit"] = state["q1"] * (state["price"] - state["marginal_cost"])
        state["plant_output"] = state["producer_one_profit"]
        state["current_error"] = state["target"] - state["plant_output"]

        # # Specify float type for control_signal because of jax trace
        # volume_change = jnp.array(state["control_signal"], dtype=jnp.float32) + disturbance - state["Q"]
        # water_height_change = volume_change / state["A"]
        # state["H"] += water_height_change
        # # Use jnp.sqrt because of jax trace
        # state["V"] = jnp.sqrt(2 * 9.81 * state["H"])
        # state["Q"] = state["V"] * state["C"]
        # state["plant_output"] = state["H"]
        # state["current_error"] = state["target"] - state["plant_output"]
        return state
