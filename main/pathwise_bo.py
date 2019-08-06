import numpy as np
import itertools
from scipy.optimize import basinhopping, fsolve, minimize
from scipy.stats import norm, lognorm
from sympy import *
from sklearn.metrics.pairwise import euclidean_distances
import DIRECT
import copy
import os
from interactive_bo_general import Interactive_BO_General

N_Iteration = 10 #number of iteration of bending path
N_Pivots = 10
Gamma = 10 #hyperparameter of optimization

class Pathwise_BO(Interactive_BO_General):

    def _suggest_init_parameters(self):
        init_rand = np.array([np.zeros(self._ndim),np.ones(self._ndim)])
        init = np.array(self._bound[:,0]) + (np.array(self._bound[:,1]) - np.array(self._bound[:,0])) * init_rand
        suggested_parameters = np.array([np.linspace(i[0],i[1],N_Pivots) for i in init.T]).T
        return suggested_parameters 


    def _suggest_next_parameters(self):
        def obj_sd(x,data):
            return (-self._sd([x]),0)

        #if self._last_selected == []:
            #set one end as last_selected input
        #    x_max = self._x[np.argmax(self.fMAP)]
        #else:
        x_max = self._last_selected[0]

        x_argmax_sd, _, _ = DIRECT.solve(obj_sd, self._bound[:,0], self._bound[:,1],
                                 maxf=1000, algmethod=1)

        pivots = np.array([np.linspace(x_max[i],x_argmax_sd[i],N_Pivots) for i in range(len(x_max))]).T

        suggested_parameters = list(self._bend(pivots))
        print("hello")
        return suggested_parameters 

    def _propose_negative_representives(self, suggested_parameters):
        if(self.fMAP == []):
            negative_representives = [suggested_parameters[0], suggested_parameters[-1]]
        else:
            ei_max = np.argmax(self.expected_improvement(self._last_selected))
            negative_representives = [suggested_parameters[0],suggested_parameters[ei_max]] #representive negative points
        return negative_representives
        

    def _bend(self, pivots):
        """ bend a path by moving pivots """
        for i in range(N_Iteration):
            for j in range(len(pivots)-2):
                j = j + 1
                F = self._d_expected_improvement([pivots[j]]).reshape(self._ndim)
                pivots[j] = pivots[j] + Gamma * F
                d = np.linalg.norm(pivots[j] - pivots[j-1])
                if d < 0.01:
                    pivots[j] = pivots[j-1] + 0.01 * (pivots[j] - pivots[j-1])
                if d > 0.4:
                    pivots[j] = pivots[j-1] + 0.4 * (pivots[j] - pivots[j-1])
                for k in range(self._ndim):
                    if pivots[j][k] < self._bound[k,0]:
                        pivots[j][k] = self._bound[k,0]
                    if pivots[j][k] > self._bound[k,1]:
                        pivots[j][k] = self._bound[k,1]
        return pivots

