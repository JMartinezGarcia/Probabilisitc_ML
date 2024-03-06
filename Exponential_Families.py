import abc
import functools

from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import norm, bernoulli, poisson
from scipy.special import gammaln

class ExponentialFamily(abc.ABC):
    
    '''
    Abstract class for Exponential Families
    
    input:
        x: input data                   |   shape   (batch_x, 1)
        params: input parameters        |   shape   (batch_params, dim_params)
        
    output:
        pdf: pdf of distribution        |   shape   (batch_x, batch_params)
    '''
        
    @abc.abstractmethod
    def log_base_measure(self, x: ArrayLike) -> ArrayLike:
        '''
        Return function log h(y)    |   shape (batch_x, 1)
        '''
        
    @abc.abstractmethod
    def log_partition_func(self, nu: ArrayLike) -> ArrayLike:
        '''
        Return function log Z(nu)   | shape (1, batch_params)
        '''
    
    @abc.abstractmethod
    def sufficient_stats(self, x: ArrayLike) -> ArrayLike:
        '''
        Return suff_stats(x)    |   shape (dim_nu ,batch_data)
        '''
    
    @abc.abstractmethod
    def natural_params(self, params: ArrayLike) -> ArrayLike:
        '''
        Return naural parameters ie nu = f(phi) | shape (dim_nu, batch_params)
        '''
        
    def log_pdf(self, x: ArrayLike, params: ArrayLike) -> ArrayLike:
        '''
        Return log pdf ie log h(x) + (nu(params).T @ suff_stats(x)) - log_partition(nu(params)) 
        
        output shape (batch_data, batch_param, dim_x)
        '''
        
        nu = self.natural_params(params)
        linear_term = self.sufficient_stats(x).T @ nu
        
        
        return self.log_base_measure(x) + linear_term - self.log_partition_func(nu)
        
        
        #linear_term = np.einsum('ij, ikl -> kjl', nu, self.sufficient_stats(x))
        
        
        
        #return self.log_base_measure(x) + linear_term - self.log_partition_func(nu)
    
    
            
            
        
    def pdf(self, x: ArrayLike, params: ArrayLike) -> ArrayLike:
        '''
        pdf = exp (log pdf)     |   shape (batch_data, dim_data, batch_param)
        '''
        return np.exp(self.log_pdf(x=x, params = params))
    
class BernoulliDist(ExponentialFamily):
    
    def sufficient_stats(self, x: ArrayLike) -> ArrayLike:
        '''
        suff_stats(x) = x   |   shape (dim_nu, batch_data)
        '''
        return x.T
    
    def log_base_measure(self, x: ArrayLike) -> ArrayLike:
        '''
        log h(x) = 0    |   shape (batch_x, 1)
        '''
        return np.zeros(x.shape[0])[:,None]
    
    def log_partition_func(self, nu: ArrayLike) -> ArrayLike:
        '''
        log Z(nu) = - log(1 + exp(nu))      |   shape   (1, batch_params)
        '''
        return np.log(1 + np.exp(nu))
    
    def natural_params(self, params: ArrayLike) -> ArrayLike:
        '''
        nu = log(params / (1 - params))     |   shape (dim_nu, batch_params)
        '''
        return np.log(params / (1 - params))
    
    
class GaussianDist(ExponentialFamily):
    
    def sufficient_stats(self, x: ArrayLike) -> ArrayLike:
        '''
        suff_stats(x) = [x,x**2]    |   shape (dim_nu, batch_data)
        '''
        return np.array([x, x**2]).squeeze()
    
    def log_base_measure(self, x: ArrayLike) -> ArrayLike:
        '''
        log h(x) = log(1 / sqrt(2 * np.pi))     |   shape (batch_x, 1)
        '''
        return np.zeros(x.shape[0])[:,None] - 0.5 * np.log(2 * np.pi)
    
    def natural_params(self, params: ArrayLike) -> ArrayLike:
        '''
        params = [nu, sigma**2]
        
        nu = [nu / sigma**2, -1 / (2 * sigma**2)]
        
        shape (dim_nu, batch_params)
        '''
        return np.array([params[0] / params[1], - 0.5 / (params[1])])
    
    def log_partition_func(self, nu: ArrayLike) -> ArrayLike:
        '''
        log Z(nu) = - nu[0]**2 / (4 * nu[1]) - 0.5 * log(-2 * nu[1])
        
        shape   (dim_nu, batch_params)
        '''
        
        return - np.array([(nu[0]**2 / (4 * nu[1])) + (0.5 * np.log(-2 * nu[1]))])
    
    
class PoissonDist(ExponentialFamily):
    
    def sufficient_stats(self, x: ArrayLike) -> ArrayLike:
        '''
        suff_stats(x) = x   |   shape (dim_nu, batch_data)
        '''
        
        return x.T
    
    def log_base_measure(self, x: ArrayLike) -> ArrayLike:
        '''
        log h(x) = -np.log(x!)     |   shape (batch_x, 1)
        '''
        return - gammaln(x+1)
    
    def natural_params(self, params: ArrayLike) -> ArrayLike:
        '''
        nu = log(params)    |   shape (dim_nu, batch_params)
        '''
        
        return np.log(params)
    
    def log_partition_func(self, nu: ArrayLike) -> ArrayLike:
        '''
        log Z(nu) = exp(nu)     |    shape   (dim_nu, batch_params)
        '''
        
        return np.exp(nu)
        
class MVGaussianDist(ExponentialFamily):
    
    def sufficient_stats(self, x: ArrayLike) -> ArrayLike:
        '''
        suff_stats(x) = [x, x @x .T]
        '''
        return np.array([x,x @ x.T])
    
    def log_base_measure(self, x: ArrayLike) -> ArrayLike:
        '''
        log h(x) = (2 * pi) ** (-D/2)
        '''
        return (2 * np.pi) ** (-x.shape[-1] / 2)
    
    def log_partition_func(self, nu: ArrayLike) -> ArrayLike:
        '''
        log Z(nu) = - 
        '''
 
 
if __name__ == "__main__":
            
    gaussian = GaussianDist()
    poi = PoissonDist()
    bern = BernoulliDist()
    
    
    params_gauss = np.array([[5.,1.],[4.,20.]]).T # shape (2,2) (batch_param,dim_nu)
    params_bern = np.array([0.8,0.7])[None,:] # shape (1,2)
    params_poi = np.array([0.4,0.5,0.6])[None,:] # shape (1,3)
    
    
    
    data_gauss = np.arange(4,7)[:,None] # shape (10,1)
    data_bern = np.array([0,1,1,0,1])[:,None] # shape (5,1)
    data_poi = np.array([4.,5.,6.])[:,None] # shape (3,1)
    
    
    np.testing.assert_allclose(bernoulli.pmf(data_bern,params_bern),
                               bern.pdf(x=data_bern,params=params_bern))
    
    
    np.testing.assert_allclose(gaussian.pdf(x=data_gauss,params=params_gauss),
                               norm.pdf(data_gauss,loc=params_gauss[0],scale = np.sqrt(params_gauss[1])))
    
    
    np.testing.assert_allclose(poi.pdf(x=data_poi, params=params_poi), 
                               poisson.pmf(k = data_poi, mu = params_poi))
    
    print("Tests Passed!!!")
    