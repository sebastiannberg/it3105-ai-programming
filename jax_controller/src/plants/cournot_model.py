from plants.base_plant import BasePlant


class CournotModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        if self.check_valid_init_state(init_plant_state):
            self.init_plant_state = init_plant_state

    def check_valid_init_state(self, init_plant_state):
        required_keys = ("max_price", "marginal_cost", "target", "q1", "q2", "q", "price", "producer_one_profit")
        if not all(key in init_plant_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in init_plant_state]
            raise ValueError(f"Missing keys in init_plant_state: {missing_keys}")
        self.enforce_q_values(init_plant_state)
        return True
    
    def enforce_q_values(self, state):
        if not (0 <= state["q1"] <= 1) or not (0 <= state["q2"] <= 1):
            raise ValueError(f"q1 and q2 can't be <0 or >1, but was {state['q1']} and {state['q2']}")
    
    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        state["q1"] += state["control_signal"]
        state["q1"] = max(0.0, min(state["q1"], 1.0))
        state["q2"] += disturbance
        state["q2"] = max(0.0, min(state["q2"], 1.0))
        state["q"] = state["q1"] + state["q2"]
        state["price"] = state["max_price"] - state["q"]
        state["producer_one_profit"] = state["q1"] * (state["price"] - state["marginal_cost"])
        state["plant_output"] = state["producer_one_profit"]
        state["current_error"] = state["target"] - state["plant_output"]
        self.enforce_q_values(state)
        return state
