import numpy as np
from scipy.stats import multivariate_normal
from utils import * 
from scipy.integrate import solve_ivp

class sim(): 
    def __init__(self, params, convert_params):
        self.J, B = convert_params(params)
        self.rv =  multivariate_normal([0, 0], B)
        
    def evolve(self, T, dt, n_frames, n_repeats):
        results = [] 
        for n in range(n_repeats): 
            y = np.zeros((2))
            res = np.zeros((n_frames, 2))
            for i in range(n_frames):
                for j in range(int(T/n_frames/dt)):
                    self._update(y, dt)
                res[i] = np.copy(y)
            results.append(res.T) 
        return results 


    def _update(self, y, dt):
        det = self.J @ y * dt 
        sto = np.sqrt(dt)*self.rv.rvs()
        y += det + sto
        

class FHN: 
    
    def __init__(self, params, convert_params, m_err=[0.01, 0.07]):
        '''
        params: parameters for the simulation
        convert_params: J, B = convert_params(params) 
        m_err: measurement errors 
        '''
        self.J, B = convert_params(params)
        self.rv =  multivariate_normal([0, 0], B)
        self.b = params[4]
        self.rvm =  multivariate_normal([0, 0], np.diagflat(np.array(m_err)**2))

    def _update(self, y, dt):
        det = self.J @ y 
        det[1] -= self.b*y[1]**3 
        sto = np.sqrt(dt) * self.rv.rvs()
        y += det*dt + sto 
        
    def evolve(self, T, dt, n_frames, n_repeats):
        results = [] 
        for n in range(n_repeats): 
            y = np.zeros((2))
            res = np.zeros((n_frames, 2))
            for i in range(n_frames):
                for j in range(int(T/n_frames/dt)):
                    self._update(y, dt)
                # add measurement error 
                res[i] = np.copy(y)+self.rvm.rvs() 
            results.append(res.T) 
        return results 
    
class FHN_det: 
    
    def __init__(self, params, get_J): 
        self.J = get_J(params) 
        self.b = params[4]
        
    def evolve(self, T, n_frames, init):
        t_eval = np.linspace(0, T, int(n_frames))
        det_res = solve_ivp(self._rhs, [0, T], init, method='LSODA', t_eval=t_eval)
        return det_res.y
        
    def _rhs(self, t, y): 
        rhs = self.J @ y 
        rhs[1] -= self.b*y[1]**3 
        return rhs
        
        