import numpy as np


class Consys:

    def __init__(self, controller, plant, epochs, timesteps, disturbance_range) -> None:
        self.controller = controller
        self.plant = plant
        self.epochs = epochs
        self.timesteps = timesteps
        self.disturbance_range = disturbance_range

    def run(self):
        print("running consys")
        self.controller.init_params()
        for _ in range(self.epochs):
            # init any other controller variables (error hisotry etc.)
            self.controller.init_error_history()
            # reset plant to initial state
            self.plant.reset_plant()
            disturbance_vector = np.random.uniform(*self.disturbance_range, size=self.timesteps)
            for _ in range(self.timesteps):
                # update plant
                plant_output = self.plant.update()
                # update controller
                control_signal = self.controller.update(plant_output)
                # save the error E for this timestep in an error history
            # compute MSE over the error history
            # compute the gradients (derivative of MSE over params (three k values or weights))
            # update params based on gradient