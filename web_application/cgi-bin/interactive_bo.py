import numpy as np
import itertools
import DIRECT
from scipy.optimize import fsolve, minimize
from scipy.stats import norm, lognorm
from sympy import *
from sklearn.metrics.pairwise import euclidean_distances
import copy
import os

Theta1 = 0.35 #hyperparameter of kernel
Theta2 = 0.005 #hyperparameter of kernel
N_Iteration = 10 #number of iteration of bending path
Gamma = 10 #hyperparameter of optimization
Gran = 10 #hyperparameter of optimization

class Interactive_BO():

    def __init__(self, bound, filepath, mode="linewise", sgm=1, theta_fixed=True):
        self._theta_fixed = theta_fixed
        self._bound = np.array(bound)
        assert len(self._bound.shape) == 2
        assert sgm > 0
        self._ndim = self._bound.shape[0]
        self._x = []  #already presented parameters
        self._u = []  #where is selected parameter in x
        self._v = []  #where is not selected parameter in x
        self._sgm = sgm #hyperparameter
        self._theta = np.ones(self._ndim+2)*Theta1 #hyperparameter
        self._theta[self._ndim:] = Theta2 #hyperparameter
        self._kernel = RBF(self._ndim)
        self._filepath = filepath
        self._mode = mode

    
    def query(self, last_selected=[], N=5):
        self._x = []
        self._u = []
        self._v = []
        pll = []
        qll = []
        #get data from a history file
        if not os.path.exists(self._filepath):
            print("No history file")
            if self._mode == "setwise":
                candidates = self._init_candidates_setwise(N)
            elif self._mode == "linewise":
                candidates = self._init_candidates_linewise()
            elif self._mode == "pathwise":
                candidates = self._init_candidates_pathwise(N)
            else:
                print("mode"+ self._mode +" is not implemented!")
                raise 
        else:
            try:
                with open(self._filepath) as f:
                    lines=f.readlines()
                for line in lines:
                    line = line.split('>')
                    pl = line[0].split(' ')
                    ql = line[1].split(' ')
                    pl = [np.array([float(i) for i in p.strip('[]').split(',')]) for p in pl]
                    ql = [np.array([float(i) for i in q.strip('[]\n').split(',')]) for q in ql]
                    for p in pl:
                        if self._x == [] or not np.any(np.all(self._x == p,axis=1)):
                            self._x.append(p)
                    for q in ql:
                        if self._x == [] or not np.any(np.all(self._x == q,axis=1)):
                            self._x.append(q)
                    self._x = list(map(np.array, set(map(tuple, self._x))))
                    pll.append(pl)
                    qll.append(ql)

            except:
                print("Invalid history file") 
                raise

            else:
                for i in range(len(pll)):
                    pl = pll[i]
                    ql = qll[i]
                    for p in pl:
                        self._u.append(np.where(np.all(self._x==p,axis=1))[0][0])
                        self._v.append([np.where(np.all(self._x==q,axis=1))[0][0] for q in ql])

                self._klu =np.linalg.cholesky(self._kernel(self._x))
                self._klu_t =np.linalg.cholesky(self._kernel(self._x).T)

                #estimate hyperparameter and latent utility function 
                self._kernel._X = self._x
                self._maximize_posterior()
                print("self._fMAP",self._fMAP)
                print("self._theta",self._theta)
                self._kfmap = np.linalg.solve(self._kernel(self._x),self._fMAP)
                if self._mode == "setwise":
                    candidates = self._candidates_setwise(N, last_selected)
                elif self._mode == "linewise":
                    candidates = self._candidates_linewise(last_selected)
                elif self._mode == "pathwise":
                    candidates = self._candidates_pathwise(N, last_selected)
                else:
                    print("mode"+ self._mode +" is not implemented!")
                    raise 
        return candidates

    def _init_candidates_setwise(self, N):
        candidates = np.array(self._bound[:,0]) + (np.array(self._bound[:,1]) - np.array(self._bound[:,0])) * np.random.rand(N,self._ndim) 
        return candidates
 
    def _init_candidates_linewise(self):
        init_rand = np.array([np.zeros(self._ndim),np.ones(self._ndim)])
        init = np.array(self._bound[:,0]) + (np.array(self._bound[:,1]) - np.array(self._bound[:,0])) * init_rand
        candidates = [init[0],init[1]]
        return candidates

    def _init_candidates_pathwise(self, N):
        init_rand = np.array([np.zeros(self._ndim),np.ones(self._ndim)])
        init = np.array(self._bound[:,0]) + (np.array(self._bound[:,1]) - np.array(self._bound[:,0])) * init_rand
        candidates = np.array([np.linspace(i[0],i[1],N) for i in init.T]).T
        return candidates

    def _candidates_setwise(self, N, last_selected):
        """ Return set of candidates.  """ 
        candidates = last_selected 
        x_swap = copy.deepcopy(self._x)
        f_swap = copy.copy(self._fMAP)
        for i in range(N-len(last_selected)):
            self._kfmap = np.linalg.solve(self._kernel(self._x),self._fMAP)
            self._klu =np.linalg.cholesky(self._kernel(self._x))
            self._klu_t =np.linalg.cholesky(self._kernel(self._x).T)
            x_ie = self._argmax_expected_improvement() + 0.001*np.random.rand() 
            self._fMAP = np.append(self._fMAP,self._mean([x_ie]))
            self._x.append(x_ie) 
            candidates = np.vstack((candidates,np.array([x_ie])))

        self._x = copy.deepcopy(x_swap)
        self._fMAP = copy.copy(f_swap)
        self._klu =np.linalg.cholesky(self._kernel(self._x))
        self._klu_t =np.linalg.cholesky(self._kernel(self._x).T)
        self._kfmap = np.linalg.solve(self._kernel(self._x),self._fMAP)
        return candidates 

    def _candidates_linewise(self, last_selected):
        x_ei = self._argmax_expected_improvement()
        """if np.any(np.sum((x_ie-np.array(self._x))**2,axis=1) < 1e-5):
            raise Exception("Maximum expected improvement is duplicated") """
        if last_selected == []:
            #set one end as last_selected input
            x_max = self._x[np.argmax(self._fMAP)]
        else:
            x_max = last_selected[0]
        candidates = [x_max,x_ei]
        return candidates 

    def _candidates_pathwise(self, N, last_selected):
        def obj_sd(x,data):
            return (-self._sd([x]),0)
        if last_selected == []:
            #set one end as last_selected input
            x_max = self._x[np.argmax(self._fMAP)]
        else:
            x_max = last_selected[0]

        # use DIRECT algorithm
        x_argmax_sd, _, _ = DIRECT.solve(obj_sd, self._bound[:,0], self._bound[:,1],
                                 maxf=1000, algmethod=1)
	

        pivots = np.array([np.linspace(x_max[i],x_argmax_sd[i],N) for i in range(len(x_max))]).T

        pivots = self._bend(pivots)
        return list(pivots)

    def _log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        frac = np.exp(f[np.array(self._u)]/self._sgm)
        denomi = np.zeros(len(self._v))
        for i in range(len(self._v)):
            denomi[i] += np.sum(np.exp(f[np.array(self._v)[i]]/self._sgm))
        denomi += np.exp(f[np.array(self._u)]/self._sgm)
        return np.sum(np.log(frac/denomi))

    def _d_log_likelihood(self, f=None):
        d_ll = np.zeros(len(f)+self._ndim+2)
        for i, fi in enumerate(f):
            for j in range(len(self._u)):
                pat2 = -np.exp(fi/(self._sgm)) / (self._sgm * (np.sum(np.exp(f[np.array(self._v[j])]))+np.exp(f[self._u[j]])))
                pat1 = 1 / self._sgm + pat2
                if i == self._u[j]:
                    d_ll[i] += pat1
                elif np.any(i==np.array(self._v[j])):
                    d_ll[i] += pat2
        return d_ll


    def _log_prior_theta(self, theta):
        if theta is None: theta = self._theta
        lnd1 = lognorm.pdf(theta[0:self._ndim],0.1,loc=np.log(0.5))
        lnd2 = lognorm.pdf(theta[self._ndim:],0.1,loc=np.log(0.005))
        return np.sum(np.log(lnd1)) + np.sum(np.log(lnd2))

    def _d_log_prior_theta(self, theta):
        if theta is None: theta = self._theta
        d_lpt = np.zeros(len(self._x)+len(theta))
        loc = np.append(np.ones(self._ndim)*np.log(0.5),np.ones(2)*np.log(0.005))
        d_lpt[len(self._x):] = -1/theta + (-1/theta) * (np.log(theta)-loc)/((0.1)**2)
        return d_lpt

    def _log_prior(self, f, theta):
        if f is None: f = self._fMAP
        if self._theta_fixed == True:
            alpha = np.linalg.solve(self._klu,f)
            return -0.5*np.sum(alpha**2)
        else:
            alpha = np.linalg.solve(np.linalg.cholesky(self._kernel(X=self._x,theta=theta)),f) 
            return -0.5*np.sum(alpha**2) + self._log_prior_theta(theta)

    def _d_log_prior_fix(self, f, theta):
        alpha = np.linalg.solve(self._klu,f)
        return -alpha 

    def _d_log_prior(self, f, theta):
        if f is None: f = self._fMAP
        d_lp = np.zeros(len(f)+len(theta))
        alpha = np.linalg.solve(np.linalg.cholesky(self._kernel(X=self._x,theta=theta)),f) 
        d_lp[0:len(f)] = -alpha
        for k in range(len(theta)):
            kernel = self._kernel(X=self._x,theta=theta)
            if k < self._ndim:
                d_kernel = np.array([[theta[self._ndim] * np.exp(-np.sum((np.array(i)-np.array(j))**2/(2*theta[0:self._ndim]**2))) * (np.array(i)[k]-np.array(j)[k])**2/(theta[k]**3)  for i in self._x] for j in self._x])
            elif k == self._ndim:
                d_kernel = np.array([[np.exp(-np.sum((np.array(i)-np.array(j))**2/(2*theta[0:self._ndim]**2)))  for i in self._x] for j in self._x])
            else:
                d_kernel = np.eye(len(f))
            d_lp[len(f)+k] = 0.5 * np.matmul(f, np.linalg.solve(kernel, np.matmul(d_kernel, np.linalg.solve(kernel,f))) )
            d_lp[len(f)+k] += -0.5 * np.trace(np.matmul(np.linalg.inv(kernel),d_kernel))
        if self._theta_fixed == True:
            return d_lp 
        else:
            return d_lp + self._d_log_prior_theta(theta)

    def _unnorm_log_posterior_fix(self, f):
        if f is None: f = self._fMAP
        return -self._log_likelihood(f) - self._log_prior(f,self._theta)

    def _unnorm_log_posterior(self, f, theta):
        if f is None: f = self._fMAP
        return -self._log_likelihood(f) - self._log_prior(f,theta)

    def _d_unnorm_log_posterior_fix(self, f):
        if f is None: f = self._fMAP
        return -self._d_log_likelihood(f)[:len(f)] - self._d_log_prior_fix(f,self._theta)

    def _d_unnorm_log_posterior(self, f, theta):
        if f is None: f = self._fMAP
        return -self._d_log_likelihood(f) - self._d_log_prior(f,theta)

    def _unnorm_log_posterior_sub_fix(self, X=None):
        return self._unnorm_log_posterior_fix(X[0:len(self._x)]) 

    def _unnorm_log_posterior_sub(self, X=None):
        return self._unnorm_log_posterior(X[0:len(self._x)],X[len(self._x):]) 

    def _d_unnorm_log_posterior_sub_fix(self, X=None):
        return self._d_unnorm_log_posterior_fix(X[0:len(self._x)])

    def _d_unnorm_log_posterior_sub(self, X=None):
        return self._d_unnorm_log_posterior(X[0:len(self._x)],X[len(self._x):])

    def _argmax_posterior_fix(self):

        init = np.zeros(len(self._x))
        bns = [(-5,5)]*(len(self._x))
        self._klu = np.linalg.cholesky(self._kernel(X=self._x,theta=self._theta)) 
        opt = minimize(self._unnorm_log_posterior_sub_fix,init,method='L-BFGS-B', bounds=bns, jac=self._d_unnorm_log_posterior_sub_fix, options={'maxiter':100})
        return opt.x 

    def _argmax_posterior(self):

        init = np.zeros(len(self._x)+self._ndim+2)
        init[len(self._x):] = np.ones(self._ndim+2)*0.1 
        bns = [(-5,5)]*(len(self._x)+self._ndim+2)
        bns[len(self._x):] = [(0.000001,None)]*(self._ndim+2)
        opt = minimize(self._unnorm_log_posterior_sub,init,method='L-BFGS-B', bounds=bns, jac=self._d_unnorm_log_posterior_sub, options={'maxiter':100})
        return opt.x 

    def _maximize_posterior(self):

        if self._theta_fixed == True:
            opt = self._argmax_posterior_fix()
            self._fMAP = opt
            self._kernel._theta = self._theta
        else:
            opt = self._argmax_posterior() #predicted value at observed points
            self._fMAP = opt[0:len(self._x)]
            self._theta = opt[len(self._x):]
            self._kernel._theta = self._theta


    def _mean(self, x):
        ks = self._kernel(x,self._x)
        return np.matmul(ks.T, self._kfmap)

    def _d_mean(self,x):
        ks = self._kernel(x,self._x)
        ks_d =  np.array([[[0 if np.all(i==j) else self._theta[self._ndim] * (-(i[k]-j[k])/self._theta[k]**2) * np.exp(-np.sum((np.array(i)-np.array(j))**2/(2*self._theta[0:self._ndim]**2))) for k in range(self._ndim)]  for i in x] for j in self._x])
        ks_d = ks_d.transpose()
        return np.matmul(ks_d, self._kfmap).transpose()

    def _sd(self, x):
        kss = self._kernel(x,x)
        ks = self._kernel(x,self._x)
        sd = np.diag(kss) - np.sum(np.linalg.solve(self._klu,ks)**2,axis=0)
        return sd

    def _co_sd(self, x):
        """ standard co-deviation """
        kss = self._kernel(x,x)
        ks = self._kernel(x,self._x)
        sol = np.linalg.solve(self._klu,ks)
        sd = kss - np.matmul(sol.T,sol)
        return sd

    def _d_sd(self, x):
        ks = self._kernel(x,self._x)
        ks_d =  np.array([[[0 if np.all(i==j) else self._theta[self._ndim] * (-(i[k]-j[k])/self._theta[k]**2) * np.exp(-np.sum((np.array(i)-np.array(j))**2/(2*self._theta[0:self._ndim]**2))) for k in range(self._ndim)]  for i in x] for j in self._x])
        ks_d = ks_d.transpose()
        sd_d = - np.matmul(ks_d, (np.linalg.solve(self._klu.T,np.linalg.solve(self._klu,ks))+np.linalg.solve(self._klu_t.T,np.linalg.solve(self._klu_t,ks)))).transpose()
        return sd_d


    def expected_improvement(self,x):
        mean_max = np.max(self._fMAP)
        mean = self._mean(x)
        sd = self._sd(x)
        ind = sd > 0
        cdf = np.zeros(mean.shape)
        cdf[ind] = norm.cdf((mean_max-mean[ind])/sd[ind])
        x_ei = (mean-mean_max)*cdf+sd*cdf
        return x_ei

    def _d_expected_improvement(self,x):
        mean_max = np.max(self._fMAP)
        mean = self._mean(x)
        d_mean = self._d_mean(x)
        sd = self._sd(x)
        d_sd = self._d_sd(x)
        ind = sd > 0
        cdf = np.zeros(mean.shape)
        pdf = np.zeros(mean.shape)
        cdf[ind] = norm.cdf((mean_max-mean[ind])/sd[ind])
        pdf[ind] = norm.pdf((mean_max-mean[ind])/sd[ind])
        d_x_ei = d_mean*cdf + (mean-mean_max)*pdf + d_sd*cdf + sd*pdf
        return d_x_ei

    def _argmax_expected_improvement(self):
        def obj(x,data):
            return (-self.expected_improvement([x]),0)

        # use DIRECT algorithm
        x, _, _ = DIRECT.solve(obj, self._bound[:,0], self._bound[:,1],
                                 maxf=1000, algmethod=1)

        return x

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


class RBF():
    """ ARD squered exponential kernel """
    def __init__(self, dim, X=None, Y=None, theta=None):
        self._ndim = dim 

        if X is None: self._X = None
        else: self._X = np.array(X)

        if Y is None: self._Y = None
        else: self._Y = np.array(Y)

        if theta is None: self._theta = np.ones(self._ndim+2) * 0.1 
        else: self._theta = theta 

    def __call__(self, X=None, Y=None, theta=None):
        if X is None and Y is None and theta is None: return self._K
        if X is None: X = self._X
        if Y is None: Y = self._Y
        if Y is None: Y = X 
        if theta is None: theta = self._theta
   
        self._ssd = np.array([[theta[self._ndim] + theta[self._ndim+1] if np.all(i==j) else theta[self._ndim] * np.exp(-np.sum((np.array(i)-np.array(j))**2/(2*theta[0:self._ndim]**2)))  for i in X] for j in Y])

        return self._ssd 
