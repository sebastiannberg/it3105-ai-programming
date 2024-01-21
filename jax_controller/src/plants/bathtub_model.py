from plants.base_plant import BasePlant

import jax.numpy as jnp


class BathtubModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        self.init_plant_state = init_plant_state
    
    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        print("CONTROL SIGNAL INSIDE PLANT", state["control_signal"])
        # print(state["control_signal"].astype(float))
        volume_change = jnp.array(state["control_signal"], dtype=jnp.float32) + disturbance - state["Q"]
        print(volume_change)
        water_height_change = volume_change / state["A"]
        state["H"] += water_height_change
        state["V"] = jnp.sqrt(2 * 9.81 * state["H"])
        state["Q"] = state["V"] * state["C"]
        state["plant_output"] = state["H"]
        state["current_error"] = state["target"] - state["plant_output"]
        return state

    # def reset_plant(self):
        # self.H = self.target
        # # TODO trengs disse to her?
        # self.V = sqrt(2 * 9.81 * self.H)
        # self.Q = self.V * self.C
        # pass
