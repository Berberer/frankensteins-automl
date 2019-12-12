import numpy as np

from math import log, ceil


# Adapted from https://github.com/zygmuntz/hyperband


class HyperbandRun:
    def __init__(self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 81  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1


# can be called multiple times
def run(self, skip_last=0):

    for s in reversed(range(self.s_max + 1)):

        # initial number of configurations
        n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

        # initial number of iterations per config
        r = self.max_iter * self.eta ** (-s)

        # n random configurations
        T = [self.get_params() for i in range(n)]

        for i in range((s + 1) - int(skip_last)):  # changed from s + 1

            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations

            n_configs = n * self.eta ** (-i)
            n_iterations = r * self.eta ** (i)

            val_losses = []
            early_stops = []

            for t in T:

                self.counter += 1

                result = self.try_params(t)

                loss = result["loss"]
                val_losses.append(loss)

                early_stop = result.get("early_stop", False)
                early_stops.append(early_stop)

                # keeping track of the best result so far (for display only)
                # could do it be checking results each time, but hey
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_counter = self.counter

                result["counter"] = self.counter
                result["params"] = t
                result["iterations"] = n_iterations

                self.results.append(result)

            # select a number of best configurations for the next loop
            # filter out early stops, if any
            indices = np.argsort(val_losses)
            T = [T[i] for i in indices if not early_stops[i]]
            T = T[: int(n_configs / self.eta)]

    return self.results
