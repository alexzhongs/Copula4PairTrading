# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @StartTime: 2023/07/17 16:35:27
# @EndTime: 2023/08/01 23:35:27
# Author: Zhong, Xianqing

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brute
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.metrics import mean_squared_error
from statsmodels.distributions.empirical_distribution import ECDF
from Copula import Copula
# from sko.GA import GA
# from scipy.optimize import differential_evolution, basinhopping, brute

#-----------------------------------------------------

def hedging_ratio_b_a(p1, p2):
    if len(p1) != len(p2):
        print("len of the two series should be same")
    
    # 动态跟踪配对rolling regression, 静态分析spread
    # 创建RollingOLS对象
    
    endog = p2
    # # with interception
    # exog = sm.add_constant(p1)
    # without interception
    exog = p1
    
    rolling_ols = RollingOLS(endog=endog, exog=exog, window=len(p1))

    # 执行滚动回归
    rolling_res = rolling_ols.fit()
    # 获取回归系数
    coefficients = rolling_res.params
    return coefficients

#----------------------------------------------------------

def gaussian_copula(uu, vv, rho):
    return Copula.gaussian_copula(uu, vv, rho)

def gaussian_copula_pdf(uu, vv, rho):
    return Copula.gaussian_copula_pdf(uu, vv, rho)

def gaussian_copula_u_v(uu, vv, rho):
    return Copula.gaussian_copula_u_v(uu, vv, rho)

def gaussian_copula_v_u(uu, vv, rho):
    return Copula.gaussian_copula_v_u(uu, vv, rho)
    

def student_copula(uy, vv, lambda_t, rho):
    '''
    首先给出二维 Student-t Copula 的条件概率, lambda_t取值(0,inf)为相依系数, rho取值(-1, 1)
    分别为 Student-t Copula 函数的自由度参数和相依参数
    '''
    pass

def gumbel_copula(uu, vv, alpha):
    return Copula.gumbel_copula(uu, vv, alpha)

def gumbel_copula_u_v(uu, vv, alpha):
    return Copula.gumbel_copula_u_v(uu, vv, alpha)

def gumbel_copula_v_u(uu, vv, alpha):
    return Copula.gumbel_copula_v_u(uu, vv, alpha)

def gumbel_copula_pdf(uu, vv, alpha):
    return Copula.gumbel_copula_pdf(uu, vv, alpha)

def frank_copula(uu, vv, theta):
    return Copula.frank_copula(uu, vv, theta)

def frank_copula_u_v(uu, vv, theta):
    return Copula.frank_copula_u_v(uu, vv, theta)

def frank_copula_v_u(uu, vv, theta):
    return Copula.frank_copula_v_u(uu, vv, theta)

def frank_copula_pdf(uu, vv, theta):
    return Copula.frank_copula_pdf(uu, vv, theta)
    
def clayton_copula(uu, vv, beta):
    return Copula.clayton_copula(uu, vv, beta)

def clayton_copula_u_v(uu, vv, beta):
    return Copula.clayton_copula_u_v(uu, vv, beta)

def clayton_copula_v_u(uu, vv, beta):
    return Copula.clayton_copula_v_u(uu, vv, beta)

def clayton_copula_pdf(uu, vv, beta):
    return Copula.clayton_copula_pdf(uu, vv, beta)

def student_copula_pdf(uu, vv, lambda_t, rho):
    return Copula.student_copula_pdf(uu, vv, lambda_t, rho)

def student_copula_u_v(uu, vv, lambda_t, rho):
    return Copula.student_copula_u_v(uu, vv, lambda_t, rho)

def student_copula_v_u(uu, vv, lambda_t, rho):
    return Copula.student_copula_v_u(uu, vv, lambda_t, rho)

#-----------------------------------------------------------------

def gaussian_copula_obj(theta, uu, vv):
    obj = -np.sum(np.log(gaussian_copula_pdf(uu, vv, theta)))
    return obj

def student_copula_obj(theta, uu, vv):
    lambda_t, rho = theta[0], theta[1]
    obj = -np.sum(np.log(student_copula_pdf(uu, vv, lambda_t, rho)))
    return obj

def frank_copula_obj(theta, uu, vv):
    obj = -np.sum(np.log(frank_copula_pdf(uu, vv, theta)))
    return obj

def clayton_copula_obj(theta, uu, vv):
    obj = -np.sum(np.log(clayton_copula_pdf(uu, vv, theta)))
    return obj

def gumbel_copula_obj(theta, uu, vv):
    obj = -np.sum(np.log(gumbel_copula_pdf(uu, vv, theta)))
    return obj

#-----------------------------------------------------

def ecdf_train(xx0, yy0):
    ecdf_x = ECDF(np.append(np.array([-1, 1]), xx0))
    ecdf_y = ECDF(np.append(np.array([-1, 1]), yy0))
    uu0 = ecdf_x(xx0)
    vv0 = ecdf_y(yy0)
    return uu0, vv0

def ecdf_test(xx0, yy0, xx1, yy1):
    ecdf_x = ECDF(np.append(np.array([-1, 1]), xx0))
    ecdf_y = ECDF(np.append(np.array([-1, 1]), yy0))
    uu1 = ecdf_x(xx1)
    vv1 = ecdf_y(yy1)
    return uu1, vv1  

def copula_train(uu0, vv0):
    uu, vv = uu0, vv0
    # gaussian
    gaussian_result = minimize(gaussian_copula_obj, x0=0.92, args=(uu, vv), bounds=[(0.01,0.99)])
    # frank
    frank_result = minimize(frank_copula_obj, x0=1, args=(uu, vv), bounds=[(0.01,30)])
    # gumbel
    gumbel_result = minimize(gumbel_copula_obj, x0=2, args=(uu, vv), bounds=[(1.01, 30)])
    # clayton
    clayton_result = minimize(clayton_copula_obj, x0=2, args=(uu, vv), bounds=[(0.01,30)])
    # student: warning, two params, one int, one float
    # theta_ranges = (slice(1, 31, 1), slice(-0.99, 1, 0.05))
    # student_result = brute(student_copula_obj, theta_ranges, args=(uu, vv), 
    #                        full_output=True, finish=None)
    student_result = minimize(student_copula_obj, x0=[2, 0.0], args=(uu, vv), bounds=[(1, 20), (-0.99, 0.99)])
    
    # in the array
    result_type = np.array(["Gaussian", "Frank", "Gumbel", "Clayton", "Student"])
    result_theta = np.array([gaussian_result.x[0], frank_result.x[0], 
                             gumbel_result.x[0], clayton_result.x[0], student_result.x], dtype=object)
    # result_obj = np.array([gaussian_result.fun, frank_result.fun, gumbel_result.fun, 
    #                        clayton_result.fun, student_result[1]])
    result_obj = np.array([gaussian_result.fun, frank_result.fun, gumbel_result.fun, 
                           clayton_result.fun, student_result.fun])
    # loc of opt obj
    opt_loc = result_obj.argmin()
    # opt params
    opt_theta = result_theta[opt_loc]
    opt_type = result_type[opt_loc]
    opt_obj = result_obj[opt_loc]
    
    print(result_obj)
    
    return opt_type, opt_theta, opt_obj

def copula_test(uu1, vv1, copula_type, theta):
    
    if copula_type == "Gaussian":
        mi_a_b, mi_b_a = gaussian_copula_u_v(uu1, vv1, theta), gaussian_copula_v_u(uu1, vv1, theta)
        return mi_a_b, mi_b_a
    elif copula_type == "Frank":
        mi_a_b, mi_b_a = frank_copula_u_v(uu1, vv1, theta), frank_copula_v_u(uu1, vv1, theta)
        return mi_a_b, mi_b_a
    elif copula_type == "Gumbel":
        mi_a_b, mi_b_a = gumbel_copula_u_v(uu1, vv1, theta), gumbel_copula_v_u(uu1, vv1, theta)
        return mi_a_b, mi_b_a
    elif copula_type == "Clayton":
        mi_a_b, mi_b_a = clayton_copula_u_v(uu1, vv1, theta), clayton_copula_v_u(uu1, vv1, theta)
        return mi_a_b, mi_b_a
    elif copula_type == "Student":
        theta0, theta1 = theta[0], theta[1]
        mi_a_b, mi_b_a = student_copula_u_v(uu1, vv1, theta0, theta1), student_copula_v_u(uu1, vv1, theta0, theta1)
        return mi_a_b, mi_b_a
    else:
        print("correct copula type required")


def auto_mi(xx0, yy0, xx1, yy1):
    # https://blog.csdn.net/qq_40039731/article/details/130126800 优化器理论
    uu0, vv0 = ecdf_train(xx0, yy0)
    opt_type, opt_theta, opt_obj = copula_train(uu0, vv0)
    uu1, vv1 = ecdf_test(xx0, yy0, xx1, yy1)
    print(opt_theta)
    mi_a_b, mi_b_a = copula_test(uu1, vv1, opt_type, opt_theta)
    return mi_a_b, mi_b_a

#-----------------------------------------------------

def data_import():
    loc = '/Users/alex_zhong/Desktop/AI驱动下的量化策略构建/期货数据/JQData/'
    file_name = 'IF/IF8888.CCFX_20100416_20300101.csv'
    hs = pd.read_csv(loc+file_name)
    hs.rename(columns={hs.columns[0]: 'date'}, inplace=True)
    # 转换格式到datetime
    hs['date'] = pd.to_datetime(hs['date'], format='%Y-%m-%d %H:%M')
    
    loc = '/Users/alex_zhong/Desktop/AI驱动下的量化策略构建/期货数据/JQData/'
    file_name = 'IH/IH9999.CCFX_20150416_20300101.csv'
    sz = pd.read_csv(loc+file_name)
    sz.rename(columns={sz.columns[0]: 'date'}, inplace=True)
    # 转换格式到datetime
    sz['date'] = pd.to_datetime(sz['date'], format='%Y-%m-%d %H:%M')  
    
    df = pd.merge(sz[['date', 'close']], hs[['date', 'close']], on='date', how='left')
    df.dropna(inplace=True)
    df.set_index('date', inplace=True)
    df = df.resample('5min').last()
    df.reset_index(inplace=True)
    df.dropna(inplace=True)
    
    df['ret_x'] = np.log(df['close_x']).diff()
    df['ret_y'] = np.log(df['close_y']).diff()
    
    df.fillna(0, inplace=True)
    return df


def backtest_position(df, upper=5, lower=-5):
    for i in range(1, len(df)):
        if df.iloc[i-1]['position_x'] == 0:
            # 开仓
            if (df.iloc[i]['flag_x_y']>upper) and (df.iloc[i]['flag_y_x']<lower):
                # short x, long y
                df['position_x'].iloc[i] = -1
                df['position_y'].iloc[i] = 1
            elif (df.iloc[i]['flag_y_x']>upper) and (df.iloc[i]['flag_x_y']<lower):
                # short y, long x
                df['position_y'].iloc[i] = -1
                df['position_x'].iloc[i] = 1
            else:
                df['position_y'].iloc[i] = df.iloc[i-1]['position_y']
                df['position_x'].iloc[i] = df.iloc[i-1]['position_x']
        elif df.iloc[i-1]['position_x'] == 1:
            # 平仓
            if (df.iloc[i]['flag_y_x']<0) and (df.iloc[i]['flag_x_y']>0):
                df['position_y'].iloc[i] = 0
                df['position_x'].iloc[i] = 0  
            else:
                df['position_y'].iloc[i] = df.iloc[i-1]['position_y']
                df['position_x'].iloc[i] = df.iloc[i-1]['position_x']
        elif df.iloc[i-1]['position_x'] == -1:
            # 平仓
            if (df.iloc[i]['flag_y_x']>0) and (df.iloc[i]['flag_x_y']<0):
                df['position_y'].iloc[i] = 0
                df['position_x'].iloc[i] = 0  
            else:
                df['position_y'].iloc[i] = df.iloc[i-1]['position_y']
                df['position_x'].iloc[i] = df.iloc[i-1]['position_x']
        else:
            df['position_y'].iloc[i] = np.nan
            df['position_x'].iloc[i] = np.nan
            
    return df


#--------------------------------------------------------------------

class CopulaForPairsTrading(object):
    def __init__(self, df, lookback=20000, calibrate_period=1000):
        self.df = df
        self.lookback = int(lookback)
        self.calibrate_period = int(calibrate_period)
        
        self.u = np.zeros(len(df))
        self.v = np.zeros(len(df))
        self.uu = np.zeros(self.lookback) # margianl distribution sample to train
        self.vv = np.array(self.lookback) # margianl distribution sample to train
        
        self.mi_a_b = np.ones(len(df)) * 0.5
        self.mi_b_a = np.ones(len(df)) * 0.5
        self.heding_ratio = np.zeros(len(df))
        
        self.win_flag = 500
        
        self.fee = np.arange(0/10000, 11/10000, 1/10000) # 从万5到千1，单边，敏感性测试
    
    def dynamic_hedging_ratio(self):
        close_x = self.df['close_x'].values
        close_y = self.df['close_y'].values
        
        period_i = 0
        while (period_i * self.calibrate_period) < len(self.df):
            if (period_i * self.calibrate_period) >= self.lookback:
                coeff = hedging_ratio_b_a(
                    p1=close_x[(period_i * self.calibrate_period - self.lookback) \
                        : period_i * self.calibrate_period],
                    p2=close_y[(period_i * self.calibrate_period - self.lookback) \
                        : period_i * self.calibrate_period]
                    )
                # # with added const / interception
                # hr = coeff[~np.isnan(coeff)][1]
                # without interception
                hr = coeff[~np.isnan(coeff)][0]
                
                if ((period_i+1) * self.calibrate_period) <= len(self.df):
                    self.heding_ratio[period_i * self.calibrate_period \
                        : (period_i+1) * self.calibrate_period] = hr
                else:
                    self.heding_ratio[period_i * self.calibrate_period : len(self.df)] = hr
            period_i += 1
        
        self.df['hedging_ratio_y_x'] = self.heding_ratio
            
    def in_sample_mi(self):
        xx0 = self.df['ret_x'].values
        yy0 = self.df['ret_y'].values
        xx1 = xx0
        yy1 = yy0
        
        mi_a_b, mi_b_a = auto_mi(xx0, yy0, xx1, yy1)
        self.df['mi_x_y'], self.df['mi_y_x'] = mi_a_b, mi_b_a 
                       
    
    def dynamic_calculate_mi(self):
        ret_x = self.df['ret_x'].values
        ret_y = self.df['ret_y'].values
        
        period_i = 0
        # loc0 + calibrate_period = loc1
        # loc0 - lookback = loc2
        # loc2 <---- lookback -----> loc0 <---- calibrate_period ----> loc1
        loc0 = period_i * self.calibrate_period
        loc1 = loc0 + self.calibrate_period
        loc2 = loc0 - self.lookback
        while loc0 < len(self.df):
            if (loc0 >= self.lookback) and (loc1 <= len(self.df)):
                xx0, yy0, xx1, yy1 = ret_x[loc2 : loc0], ret_y[loc2 : loc0],\
                    ret_x[loc0 : loc1], ret_y[loc0 : loc1]
                mi_a_b, mi_b_a = auto_mi(xx0, yy0, xx1, yy1)
                self.mi_a_b[loc0 : loc1] = mi_a_b
                self.mi_b_a[loc0 : loc1] = mi_b_a
            elif (loc0 >= self.lookback) and (loc1 > len(self.df)):
                xx0, yy0, xx1, yy1 = ret_x[loc2 : loc0], ret_y[loc2 : loc0],\
                    ret_x[loc0 : len(self.df)], ret_y[loc0 : len(self.df)]
                mi_a_b, mi_b_a = auto_mi(xx0, yy0, xx1, yy1)
                self.mi_a_b[loc0 : len(self.df)] = mi_a_b
                self.mi_b_a[loc0 : len(self.df)] = mi_b_a               
            
            period_i += 1
            loc0 = period_i * self.calibrate_period
            loc1 = loc0 + self.calibrate_period
            loc2 = loc0 - self.lookback        
            
        self.df['mi_x_y'], self.df['mi_y_x'] = self.mi_a_b, self.mi_b_a  
        
    def dynamic_calculate_flag(self):
        self.df['cumsum_mi_x_y'] = (self.df['mi_x_y'] - 0.5).cumsum()
        self.df['cumsum_mi_y_x'] = (self.df['mi_y_x'] - 0.5).cumsum()
        self.df['flag_x_y'] = self.df['cumsum_mi_x_y'].diff(self.win_flag)
        self.df['flag_y_x'] = self.df['cumsum_mi_y_x'].diff(self.win_flag)
        
            
    def simulate_trading_param(self):
        # create long - short position
        self.df['position_x'], self.df['position_y'] = np.zeros(len(self.df)), np.zeros(len(self.df))
        # backtest position
        self.df = backtest_position(self.df)
        # create if_fee position： when 0 -> non 0, if_fee = 0
        self.df['if_fee'] = np.zeros(len(self.df))
        # 矢量化的方式计算if_fee
        self.df['if_fee'] = (
            (self.df['position_x'] != 0) & (self.df['position_y'].shift(1) == 0)
            ).astype(int)
        # 如果第一行的 'position_x' 为非0，将 'if_fee' 设置为0
        if self.df.at[0, 'position_x'] != 0:
            self.df.at[0, 'if_fee'] = 0
    
    def simulate_trading_ret(self):
        for fee in self.fee:
            self.df['pnl_ret_'+str(fee)] = (
                self.df['position_x'] * self.df['hedging_ratio_y_x']
                ).shift() * self.df['ret_x']\
                    + self.df['position_y'].shift() * self.df['ret_y']\
                        - 2 * fee * self.df['if_fee']
                        
    def simulate_trading_pnl(self):
        for fee in self.fee:
            self.df['pnl_'+str(fee)] = (1 + self.df['pnl_ret_'+str(fee)]).cumprod()
        
    def simulate_trading(self):
        self.simulate_trading_param()
        self.simulate_trading_ret()
        self.simulate_trading_pnl()
            
            
    # def dynamic_marginal_cdf(self):
    #     ret_x = self.df['ret_x'].values
    #     ret_y = self.df['ret_y'].values
                
    #     period_i = 0
    #     while (period_i * self.calibrate_period) < len(self.df):
    #         if (period_i * self.calibrate_period) >= self.lookback:
                
    #             ecdf_x = ECDF(np.append(ret_x[(period_i * self.calibrate_period - self.lookback) \
    #                 : period_i * self.calibrate_period],[-1, 1]))
    #             ecdf_y = ECDF(np.append(ret_y[(period_i * self.calibrate_period - self.lookback) \
    #                 : period_i * self.calibrate_period],[-1, 1]))
                
    #             if ((period_i+1) * self.calibrate_period) <= len(self.df):
    #                 self.uu = ecdf_x(ret_x[period_i * self.calibrate_period \
    #                     : (period_i+1) * self.calibrate_period])
    #                 self.u[period_i * self.calibrate_period \
    #                     : (period_i+1) * self.calibrate_period] = self.uu
    #                 self.vv = ecdf_y(ret_y[period_i * self.calibrate_period \
    #                     : (period_i+1) * self.calibrate_period])
    #                 self.v[period_i * self.calibrate_period \
    #                     : (period_i+1) * self.calibrate_period] = self.vv
    #             else:
    #                 self.uu = ecdf_x(ret_x[period_i * self.calibrate_period : len(self.df)])
    #                 self.vv = ecdf_y(ret_y[period_i * self.calibrate_period : len(self.df)])
    #                 self.u[period_i * self.calibrate_period : len(self.df)] = self.uu
    #                 self.v[period_i * self.calibrate_period : len(self.df)] = self.vv
    #         period_i += 1
    #     pass
        
        
        
if __name__ == "__main__":
    df = data_import()    
    pt = CopulaForPairsTrading(df)
    pt.dynamic_calculate_mi()
    # pt.in_sample_mi()
    pt.dynamic_calculate_flag()
    pt.dynamic_hedging_ratio()
    print(pt.df)
    plt.plot((pt.df.cumsum_mi_x_y-pt.df.cumsum_mi_y_x).values)

    plt.show()
    




