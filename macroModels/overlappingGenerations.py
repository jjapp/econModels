import numpy as np

class overlappingGenerations:
    '''Creates an overlapping generations model.  The default values are a constant relative risk aversion
    and a theta of 1.'''

    def __init__(self, utility='CRRA', rho=0, r=None, theta=1, g=0.05, n=0.05, periods=100, w=5, alpha=0.5):
        self.utility = utility
        self.rho = rho
        # utility function
        if utility is None:
            self.utility = 'CRRA'
        else:
            self.utility = 'CARA'

        # interest rates
        if r is None:
            self.r = 0.03
        else:
            self.r = r
        self.theta = theta
        self.g = g
        self.n = n
        self.periods = periods
        self.s = ((1 + self.r) ** ((1 - self.theta) / self.theta)) / (
                    (1 + self.rho) ** (1 / self.theta) + (1 + self.r) ** ((1 - self.theta) / self.theta))
        self.w = w  # need to change this
        self.alpha = alpha

    def get_k2(self, row):
        '''Returns second period capital based on inputs and first period capital'''
        self.u = 1 / (1 + self.n)
        self.v = 1 / (1 + self.g)
        self.new = self.u * self.v * self.s * (1 - self.alpha) * row ** self.alpha
        return self.new

    def get_model_dynamics(self):
        '''Returns a series of the system dynamics with a 2xN series with
        Kt in the first column and kt+1 on the second column'''

        k1 = np.linspace(0, 1, 100)
        self.k2 = []
        for row in k1:
            self.k2.append(self.get_k2(row))

        self.dynamics = np.column_stack((k1, self.k2))

        return self.dynamics 