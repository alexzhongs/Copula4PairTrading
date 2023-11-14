# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time: 2023/07/15 16:35:26
# Author: Zhong, Xianqing

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import t
from math import gamma

def Gumbel_copula(u, v, alpha):
    # 基于Copula理论的股票配对交易策略研究_闵济康 P17 - P20
    # alpha >= 1, =1 时uv独立
    u = np.array(u)
    v = np.array(v)

    F = np.exp(
        -(
            (-np.log(u)) ** alpha + (-np.log(v)) ** alpha
            ) ** (1 / alpha)
    )
    return F

class Copula(object):
    # 强烈建议不要使u, v = 0, 1，可能会返回nan

    def gaussian_copula(u, v, rho):
        # A+H股配对交易策略_吴孟卿 P8 - P10
        # Gaussian copula预设是二元分布
        # input: uv 为 通过ECDF估计得到的边缘分布函数值，可以是标量也可以是向量，rho为需要估计的相关系数
        # output: 输出联合分布函数值
        # C(u, v) = phi_2 (phi^{-1}(u), phi^{-1}(v))
        
        # when u,v = 1时ppf得到inf, 处理0时转为nan,可能是multi_normal cdf的问题
        
        '''
        Example:
        u = [1, 1]
        v = [1, 0.5]
        cov = 0
        
        return [1, 0.5]
        '''
        
        inv1 = norm.ppf(u) # ppf就是用来求逆，可通过norm.cdf(norm.ppf(u))来验证
        inv2 = norm.ppf(v)
        arr = np.dstack((inv1, inv2))
        
        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])
        
        multi_normal = multivariate_normal(mean=mu, cov=cov)

        return multi_normal.cdf(arr)

    def gaussian_copula_pdf(u, v, rho):
        # rho介于 -1到1 之间
        u = np.array(u)
        v = np.array(v)
        
        inv1 = norm.ppf(u)
        inv2 = norm.ppf(v)
        
        numerator = (2 * rho * inv1 * inv2) - (rho ** 2) * (inv1 ** 2 + inv2 ** 2)
        denominator = 2 * (1 - rho ** 2)
        
        f = 1 / np.sqrt(1 - rho**2) * np.exp(numerator / denominator)
        return f

    def gaussian_copula_u_v(u, v, rho):
        # 基于Copula理论的股票配对交易策略研究_闵济康 P17 - P20
        # 返回V = v时，U <= u的条件概率
        
        inv1 = norm.ppf(u) # ppf就是用来求逆，可通过norm.cdf(norm.ppf(u))来验证
        inv2 = norm.ppf(v)
        
        numerator = inv1 - rho * inv2
        denominator = np.sqrt(1 - rho ** 2)
        
        return norm.cdf(numerator / denominator)

    def gaussian_copula_v_u(u, v, rho):
        # 基于Copula理论的股票配对交易策略研究_闵济康 P17 - P20
        # 返回U = u时，V <= v的条件概率
        
        inv1 = norm.ppf(u) # ppf就是用来求逆，可通过norm.cdf(norm.ppf(u))来验证
        inv2 = norm.ppf(v)
        
        numerator = inv2 - rho * inv1
        denominator = np.sqrt(1 - rho ** 2)
        
        return norm.cdf(numerator / denominator)
        

    def student_copula(u, v, lambda_t, rho):
        '''
        首先给出二维 Student-t Copula 的条件概率, lambda_t取值(0,inf)为相依系数, rho取值(-1, 1)
        分别为 Student-t Copula 函数的自由度参数和相依参数
        '''
        pass
    
    def student_copula_pdf(u, v, lambda_t, rho):
        # https://www1feb-uva.nl/ke/act/people/kaas/MART-Sec-7.8.3-4.pdf
        
        u = np.array(u)
        v = np.array(v)
        
        x1 = t.ppf(u, lambda_t)
        x2 = t.ppf(v, lambda_t)
        
        # factor1 = 1 / (2 * np.pi * np.sqrt(1 - rho**2))
        # factor2 = 1 + 1 / (lambda_t * (1 - rho**2)) \
        #     * (x1**2 - 2*rho*x1*x2 + x2**2)
        # power = - (lambda_t + 2) / 2
        
        # pdf = factor1 * factor2 ** power
        
        factor1 = 1 / np.sqrt(1 - rho**2)
        factor2 = gamma((2+lambda_t)/2) * gamma(lambda_t/2) / (gamma((1+lambda_t)/2)) ** 2
        numerator = 1 + (x1**2 + x2**2 - 2*rho*x1*x2) / (lambda_t * (1-rho**2))
        denominator = (1 + (x1**2 / lambda_t)) * (1 + (x2**2 / lambda_t))
        power = - (1+lambda_t/2)
        factor3 = (numerator / denominator) ** power
        pdf = factor1 * factor2 * factor3
        
        return pdf
    
    def student_copula_u_v(u, v, lambda_t, rho):
        # [Quantitative Finance 2016-apr 27 vol. 16 iss. 10] 
        # Rad, Hossein_ Low, Rand Kwong Yew_ Faff, Robert -
        # The profitability of pairs trading strategies_ distance, cointegration and copula methods
        # (2016) [10.1080_14697688.2016.11643
        u = np.array(u)
        v = np.array(v)
        
        x1 = t.ppf(u, lambda_t)
        x2 = t.ppf(v, lambda_t)
        
        numerator = x1 - rho * x2
        denominator = np.sqrt((lambda_t + x2**2) * (1 - rho**2)\
            / (lambda_t + 1))
        
        F_u_v = t.cdf(numerator/denominator, lambda_t+1)
        return F_u_v

    def student_copula_v_u(u, v, lambda_t, rho):
        # [Quantitative Finance 2016-apr 27 vol. 16 iss. 10] 
        # Rad, Hossein_ Low, Rand Kwong Yew_ Faff, Robert -
        # The profitability of pairs trading strategies_ distance, cointegration and copula methods
        # (2016) [10.1080_14697688.2016.11643
        u = np.array(u)
        v = np.array(v)
        
        x1 = t.ppf(u, lambda_t)
        x2 = t.ppf(v, lambda_t)
        
        numerator = x2 - rho * x1
        denominator = np.sqrt((lambda_t + x1**2) * (1 - rho**2)\
            / (lambda_t + 1))
        
        F_v_u = t.cdf(numerator/denominator, lambda_t+1)
        return F_v_u

    def gumbel_copula(u, v, alpha):
        return Gumbel_copula(u, v, alpha)

    def gumbel_copula_u_v(u, v, alpha):
        #返回V = v时，U <= u的条件概率
        u = np.array(u)
        v = np.array(v)
        
        F_u_v = Gumbel_copula(u, v, alpha) \
            * ((-np.log(u))**alpha + (-np.log(v))**alpha) ** ((1 - alpha) / alpha) \
                * (-np.log(v)) ** (alpha - 1) \
                    * 1 / v
                    
        return F_u_v

    def gumbel_copula_v_u(u, v, alpha):
        #返回V = v时，U <= u的条件概率
        u = np.array(u)
        v = np.array(v)
        
        F_v_u = Gumbel_copula(u, v, alpha) \
            * ((-np.log(u))**alpha + (-np.log(v))**alpha) ** ((1 - alpha) / alpha) \
                * (-np.log(u)) ** (alpha - 1) \
                    * 1 / u
                    
        return F_v_u

    def gumbel_copula_pdf(u, v, alpha):
        u = np.array(u)
        v = np.array(v)
        
        factor1 = Gumbel_copula(u, v, alpha=alpha) / (u * v)
        factor2 = (np.log(u) * np.log(v)) ** (alpha-1) / ((-np.log(u))**alpha + (-np.log(v))**alpha) ** (2-1/alpha)
        factor3 = alpha - 1 + ((-np.log(u))**alpha + (-np.log(v))**alpha) ** (1/alpha)
        
        pdf = factor1 * factor2 * factor3
        
        return pdf

    def frank_copula(u, v, theta):
        # theta的取值范围在(-inf, 0) or (0, inf)之间， 当theta > 0室二者正相关， 趋向于0时相互独立
        u = np.array(u)
        v = np.array(v)
        
        if theta == 0:
            print('theta should not equals 0')
            
        numerator = (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))
        denominator = 1 - np.exp(-theta)
            
        F = - 1 / theta * np.log(
            1 - numerator / denominator
        )
        return F

    def frank_copula_u_v(u, v, theta):
        # 返回V = v时，U <= u的条件概率
        u = np.array(u)
        v = np.array(v)
        
        numerator = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta*v)-1)
        denominator = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1)
        
        F_u_v = numerator / denominator
        return F_u_v

    def frank_copula_v_u(u, v, theta):
        # 返回V = v时，U <= u的条件概率
        u = np.array(u)
        v = np.array(v)
        
        numerator = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta*u)-1)
        denominator = (np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1)
        
        F_v_u = numerator / denominator
        return F_v_u

    def frank_copula_pdf(u, v, theta):
        u = np.array(u)
        v = np.array(v)
        
        # numerator = -theta * (np.exp(-theta) - 1) * np.exp(-theta * (u+v)) 
        # denominator = ((np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(theta)-1)) ** 2
        
        # numerator = theta * (np.exp(theta) - 1) * np.exp(theta * (u+v)) 
        # denominator = ((np.exp(theta*u)-1) * (np.exp(theta*v)-1) + (np.exp(theta)-1)) ** 2
        
        numerator = -theta * (np.exp(-theta) - 1) * np.exp(-theta * (u+v)) 
        denominator = ((np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1)) ** 2
        
        pdf = numerator / denominator
        
        return pdf
        
    def clayton_copula(u, v, beta):
        # beta 取值范围(0, inf),越小越独立
        u = np.array(u)
        v = np.array(v)
        
        F = (u**(-beta) + v**(-beta) - 1) ** (-1/beta)
        return F

    def clayton_copula_u_v(u, v, beta):
        #probability U <= u, condition v
        u = np.array(u)
        v = np.array(v)
        
        f1 = u ** (-(beta+1))
        f2 = (u**(-beta) + v**(-beta) -1) ** (- 1/beta - 1)
        
        F_u_v = f1 * f2
        return F_u_v


    def clayton_copula_v_u(u, v, beta):
        #probability V <= v, condition u
        u = np.array(u)
        v = np.array(v)
        
        f1 = v ** (-(beta+1))
        f2 = (u**(-beta) + v**(-beta) -1) ** (- 1/beta - 1)
        
        F_v_u = f1 * f2
        return F_v_u

    def clayton_copula_pdf(u, v, beta):
        u = np.array(u)
        v = np.array(v)
        
        pdf = (beta+1) * (u*v)**(-beta-1) * (u**(-beta) + v**(-beta) - 1)**(-(2*beta+1)/beta)
        return pdf
    
if __name__ == "__main__":
    
    u = np.arange(0.01, 1, 0.01)
    v = np.ones(len(u)) * 0.5
    pdf = Copula.student_copula_pdf(u, v, 150, 0.7)
    pdf1 = Copula.gaussian_copula_pdf(u, v, 0.7)
    pdf2 = Copula.gumbel_copula_pdf(u, v, 1)

    print(np.sum(pdf))
    print(np.sum(pdf1))
    print(np.sum(pdf2))