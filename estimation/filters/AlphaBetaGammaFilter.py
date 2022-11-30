""" Alpha-Beta-Gamma Filter Implementation

Based on the examples:
https://www.kalmanfilter.net/alphabeta.html
"""

import math
import numpy as np


class AlphaBetaGammaFilter():
    def __init__(
        self, 
        alpha:float = 1.0,
        beta:float = 1.0,
        gamma:float = 1.0,
        delta_t:float = 1.0
        ) -> None:
        """ Alpha-Beta-Gamma Filter class for tracking
        
        Input:
            alpha: (float) multiplier for position prediction equation (0.0 <= alpha <= 1.0)
            beta: (float) multiplier for velocity prediction equation (0.0 <= beta <= 1.0)
            gamma: (float) multiplier for acceleration prediction equation (0.0 <= gamma <= 1.0)
            delta_t: (float) timing interval
        
        Return: None
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.delta_t = delta_t
        self.timer = 0
        
        # State update equation variables
        # This is for estimating the current position, velocity, and acceleration
        #  considering previous estimates and the current measurement
        self.measured_pos = 0.0 # measured position
        self.current_x = 0.0 # initial (current) position
        self.current_v = 0.0 # initial (current) velocity
        self.current_a = 0.0 # initial (current) acceleration
        
        # State extrapolation variables
        # This is for predicting position, velocity, and acceleration for next time frame
        # We use the "current" variables to predict the below attributes
        self.pred_x = 0.0
        self.pred_v = 0.0
        self.pred_a = 0.0
    

    def initialize(
        self, 
        x:float = None, 
        v:float = None, 
        a:float = None
        ) -> None:
        """ Initialize initial conditions (position, velocity, and acceleration)"""
        self.current_x = x
        self.current_v = v
        self.current_a = a
        
        # With initial conditions set, predict for an initial guess
        self.pred_x = self.current_x + self.current_v * self.delta_t + (0.5 * self.current_a * self.delta_t**2)
        self.pred_v = self.current_v + self.current_a * self.delta_t
        self.pred_a = self.current_a
        
        print('Position     (t={}): {}'.format(self.timer,self.pred_x))
        print('Velocity     (t={}): {}'.format(self.timer,self.pred_v))
        print('Acceleration (t={}): {}'.format(self.timer,self.pred_a))
    

    def update(
        self,
        z:float = None
        ) -> None:
        """ Compute:
            i)  current estimate with state update equations
            ii) next state estimate (prediction)
        """
        # Update timer first...
        self.timer += 1
        self.current_x = self.pred_x
        self.current_v = self.pred_v
        self.current_a = self.pred_a
        
        # Compute current estimate
        x = self.current_x
        diff = z - x
        self.current_x = x + self.alpha * diff
        self.current_v = self.current_v + self.beta * (diff / self.delta_t)
        self.current_a = self.current_a + self.gamma * (diff / (0.5 * self.delta_t**2))
        
        print('Position Est.     (t={}): {}'.format(self.timer,self.current_x))
        print('Velocity Est.     (t={}): {}'.format(self.timer,self.current_v))
        print('Acceleration Est. (t={}): {}'.format(self.timer,self.current_a))
        
        
        # Compute prediction for current time
        self.pred_x = self.current_x + self.current_v * self.delta_t + (0.5 * self.current_a * self.delta_t**2)
        self.pred_v = self.current_v + self.current_a * self.delta_t
        self.pred_a = self.current_a
        
        print('Position     (t={}): {}'.format(self.timer,self.pred_x))
        print('Velocity     (t={}): {}'.format(self.timer,self.pred_v))
        print('Acceleration (t={}): {}'.format(self.timer,self.pred_a))
    


if __name__ == "__main__":
    abg_filter = AlphaBetaGammaFilter(
        alpha=0.5, 
        beta=0.4, 
        gamma=0.1, 
        delta_t=5
    )
    
    abg_filter.initialize(
        x=30000,
        v=50,
        a=0
    )
    
    abg_filter.update(z=30160)
    abg_filter.update(z=30365)
    
