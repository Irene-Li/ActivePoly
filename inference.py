import numpy as np
from scipy.optimize import minimize, brute
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import multivariate_normal
import scipy

class inference():
    '''
    basic inference class that assumes a linear 6 parameter model 
    '''
    
    def __init__(self, convert_params):
        self.convert_params = convert_params 
    
    def inf(self, guess, trajs, dt, bounds=None, brute_force=False, method='trust-constr'):   
        cost = lambda x: self._minuslogP(x, trajs, dt)
        if bounds is None: 
            bounds = [[1e-10, None]]*len(guess)
            bounds[1] = [None, None]
        if brute_force: 
            res = brute(cost, bounds, Ns=100)
            MAP, cost = res[0], res[1]
        else: 
            res = minimize(cost, guess, bounds=bounds, method=method)
            MAP, cost = res.x, res.fun 
        return MAP, cost

    def std_error(self, MAP, trajs, dt, delta=1e-3):
        cost = lambda x: self._minuslogP(x, trajs, dt)
        temp = np.copy(MAP)
        if not isinstance(delta, list):
            delta = MAP*delta 
        hess = np.empty((len(MAP), len(MAP)))
        for i in range(len(MAP)):
            for j in range(i, len(MAP)):
                temp[i] += delta[i]
                temp[j] += delta[j]
                hess[i, j] = cost(temp) # ++ 
                temp[j] -= 2*delta[j]
                hess[i, j] -= cost(temp) # +- 
                temp[i] -= 2*delta[i]
                hess[i, j] += cost(temp) # -- 
                temp[j] += 2*delta[j]
                hess[i, j] -= cost(temp) # -+
                hess[i, j] /= (4*delta[i]*delta[j])
                hess[j, i] = hess[i, j]   
        return np.sqrt(np.diagonal(np.linalg.inv(hess))), hess

    def _minuslogP(self, params, trajs, dt):
        '''
        traj: 2 x T  
        '''
        J, B = self.convert_params(params)
        self._set_up(J, dt)
        cov = self._cov(J, B) 
        invcov = np.linalg.inv(cov)
        sign, norm = np.linalg.slogdet(invcov)
        
        minuslogp = 0 
        for traj in trajs: 
            T = traj.shape[-1]-1
            xm = self.evo @ traj[:, :-1]
            diff = (traj[:, 1:] - xm)
            minuslogp += np.einsum('ji,jk,ki', diff, invcov, diff).real/2 
            minuslogp -= T*norm/2
        return minuslogp

    def _set_up(self, J, dt):
        self.evo =  scipy.linalg.expm(J*dt)
        
    def _cov(self, J, B):
        c = solve_continuous_lyapunov(J, -B)  
        return - self.evo @ c @ self.evo.T + c 


class exact_inference(inference): 
    
    def _minuslogP(self, J, B, trajs, dt):
        self._set_up(J, dt)
        invcov = np.linalg.inv(self._cov(J, B)) 
        sign, norm = np.linalg.slogdet(invcov)
        
        minuslogp = 0 
        for traj in trajs: 
            T = traj.shape[-1]-1
            xm = self.evo @ traj[:, :-1]
            diff = (traj[:, 1:] - xm)
            minuslogp += np.einsum('ji,jk,ki', diff, invcov, diff).real/2 
            minuslogp -= T*norm/2
        return minuslogp
    
    def inf(self, trajs, dt):

        T1 = np.zeros((2, 2))
        T2 = np.zeros((2, 2))
        T3 = np.zeros((2, 2))
        for traj in trajs: 
            t1, t2, t3 = self._compute_suff_stats(traj) 
            T1 += t1 
            T2 += t2 
            T3 += t3 
        T3_inv = np.linalg.inv(T3)
        Q = T2 @ T3_inv
        Cov = (T1 - T2 @ T3_inv @ T2.T)/np.sum([traj.shape[-1] -1 for traj in trajs])

        J = scipy.linalg.logm(Q)/dt 
        c = scipy.linalg.solve_discrete_lyapunov(Q, Cov)
        B = - J @ c - c @ J.T 
        return J, B


    def _compute_suff_stats(self, traj):
        x_later = traj[:, 1:]
        x_earlier = traj[:, :-1]
        T1 = self._outer(x_later, x_later)
        T2 = self._outer(x_later, x_earlier)
        T3 = self._outer(x_earlier, x_earlier)
        return T1, T2, T3 

    def _outer(self, a, b): 
            return np.einsum('ij,kj->ik', a, b)
        
        
class nonlinear_inference(inference):
                   
    def _minuslogP(self, params, trajs, dt):
        '''
        traj: 2 x T 
        '''
        J, B = self.convert_params(params)
        b = params[4]
        invB = np.linalg.inv(B*dt) 
        _, norm = np.linalg.slogdet(invB)

        minuslogp = 0 
        for traj in trajs: 
            N = traj.shape[-1] - 1
            dx = traj[:, 1:] - traj[:, :-1]
            det = J @ traj[:, :-1]
            det[1] -= b*traj[1, :-1]**3 
            diff = dx - det*dt 
    
            minuslogp += np.einsum('ji,jk,ki', diff, invB, diff)/2 
            minuslogp -= N*norm/2
        return minuslogp
