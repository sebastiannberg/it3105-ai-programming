import numpy as np


class Consys:

    def __init__(self, controller, plant, epochs, timesteps, disturbance_range) -> None:
        self.controller = controller
        self.plant = plant
        self.epochs = epochs
        self.timesteps = timesteps
        self.disturbance_range = disturbance_range

    def mean_squared_error(self, error_history):
        return sum((error ** 2 for error in error_history)) / len(error_history)

    def run(self):
        print("running consys")
        self.controller.init_params()
        for _ in range(self.epochs):
            # init any other controller variables (error hisotry etc.)
            self.controller.init_error_history()
            # reset plant to initial state
            self.plant.reset_plant()
            disturbance_vector = np.random.uniform(*self.disturbance_range, size=self.timesteps)
            for t in range(self.timesteps):
                # TODO update controller before plant?
                # update plant
                error = self.plant.update(self.controller.control_signal, disturbance_vector[t])
                # update controller
                self.controller.update_control_signal(error)
                # save the error E for this timestep in an error history (maybe inside controller object)
                self.controller.save_error(error)
                print(self.plant.H)
            # compute MSE over the error history
            # TODO inside controller instead
            mse = self.mean_squared_error(self.controller.error_history)
            print(mse)
            # compute the gradients (derivative of MSE over params (three k values or weights))
            # update params based on gradient