from controllers.base_controller import BaseController
import numpy as np
import jax
import jax.numpy as jnp


class AIPIDController(BaseController):

    def __init__(self, activation_function, num_hidden_layers, neurons_per_layer, initial_weight_bias_range) -> None:
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
        if activation_function == "tanh":
            self.activation_function = self.tanh
        if activation_function == "relu":
            self.activation_function = self.relu
        if not num_hidden_layers == len(neurons_per_layer):
            raise ValueError("Number of hidden layers and neurons per layer do not correspond, check config.py")
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.initial_weight_bias_range = initial_weight_bias_range

    def init_params(self):
        params = []
        # Size of input layer set to 3 so we can model PID
        input_size = 3
        # Add weights and bias for hidden layers
        for num_neurons in self.neurons_per_layer:
            params.append(np.random.uniform(low=self.initial_weight_bias_range[0],
                                  high=self.initial_weight_bias_range[1],
                                  size=(num_neurons, input_size + 1))) # Add bias term to the input_size
            # Update input_size for next iteration of loop
            input_size = num_neurons
        # Add weights and bias for output layer (one neuron for control signal)
        params.append(np.random.uniform(low=self.initial_weight_bias_range[0],
                                   high=self.initial_weight_bias_range[1],
                                   size=(1, input_size + 1))) # Add bias term to the input_size
        return params
    
    def update_controller(self, params, state):
        if not state["error_history"]:
            # Edge case because error_history has no values
            input = jnp.array([state["current_error"], 0, state["current_error"]]).reshape(3,1)
        else:
            input = jnp.array([state["current_error"], jnp.sum(jnp.array(state["error_history"])), state["current_error"] - state["error_history"][-1]]).reshape(3,1)
        
        for layer in params:
            output = self.compute_layer_output(layer, input)
            # Update input for the next iteration
            input = output

        # Update control_signal after neural network is done
        state["control_signal"] = output[0, 0]
        return state
    
    def compute_layer_output(self, layer_params, input):
        try:
            print(layer_params.shape, input.shape)
            # Do not include bias when multiplying the weights, add it afterwards
            result = jnp.dot(layer_params[:, :-1], input) + jnp.reshape(layer_params[:, -1], (-1, 1)) # Reshape bias before adding
            # Apply activation function
            result = self.activation_function(result)
            return result

        except ValueError as e:
            print(f"Matrix multiplication error: {e}")

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        return jnp.tanh(x)

    def relu(self, x):
        return jnp.maximum(0, x)
