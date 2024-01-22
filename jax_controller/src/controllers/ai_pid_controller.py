from controllers.base_controller import BaseController
import numpy as np
import jax.numpy as jnp


class AIPIDController(BaseController):

    def __init__(self, activation_function, num_hidden_layers, neurons_per_layer, initial_weight_bias_range) -> None:
        self.activation_function = activation_function
        if not num_hidden_layers == len(neurons_per_layer):
            raise ValueError("Number of hidden layers and neurons per layer do not correspond, check config")
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.initial_weight_bias_range = initial_weight_bias_range

    def init_params(self):
        # TODO how to set these layers to params, list of lists?
        # Size of input layer set to 3 to model PID
        input_size = 3
        # Add weights and bias for hidden layers
        for hidden_layer, num_neurons in zip(range(self.num_hidden_layers), self.neurons_per_layer):
            print(hidden_layer, num_neurons)
            params = np.random.uniform(low=self.initial_weight_bias_range[0],
                                  high=self.initial_weight_bias_range[1],
                                  size=(num_neurons, input_size + 1)) # Add bias term to the input_size
            print(params.shape)
            # Update input_size for next iteration of loop
            input_size = num_neurons
        # Add weights and bias for output layer (one neuron for control signal)
        print("output layer")
        params = np.random.uniform(low=self.initial_weight_bias_range[0],
                                   high=self.initial_weight_bias_range[1],
                                   size=(1, input_size + 1)) # Add bias term to the input_size
        print(params.shape)
    
    def update_controller(self, params, state):
        if not state["error_history"]:
            # Edge case because error_history has no values
            features = jnp.array([state["current_error"], 0, state["current_error"]])
            # TODO better handling of this
            output = 1 + 1
        else:
            features = jnp.array([state["current_error"], jnp.sum(jnp.array(state["error_history"])), state["current_error"] - state["error_history"][-1]])
        return state
