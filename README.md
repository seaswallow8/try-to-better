# try-to-better
Financial Quant
hello world
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 08:42:30 2018

@author: seaswallow
"""

##############################################
##在python 中 引入 的第三方包
########
import os
os.chdir('/Users/seaswallow/Documents/try_better/python/chapter6/data')
import sys
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import scipy
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
from pandas import Series,DataFrame
from scipy.stats import t
import scipy.stats as stats
from scipy.stats import norm
import scipy.stats as st 

############################################################
####### question one  ######
####################################################

data1 = pd.read_excel('question_1.xlsx')

#### return ####
data1['return'] = np.log(data1.Close/data1['Close'].shift(1))
data1 = data1.dropna()
data1 = data1.reset_index(drop = True)
np.array(data1['return'] ) 

#### 求有关收益率的统计量特征 ##########
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt


#### 标准差  ####

sta = scs.describe(np.array(data1['return'] ) )
zt = np.array(data1['return'])/np.sqrt(sta[3])


def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
    return 

tsplot(zt, lags=30)

############################################################
####### question two  and three ######
############################################################
data2 = pd.read_excel('question_2.xlsx')

#### return ####
data2['return'] = np.log(data2.Close/data2['Close'].shift(1))
data2 = data2.dropna()
data2 = data2.reset_index(drop = True)
var   = np.array(data2['s2t']) 
data2['sqr_var'] = np.sqrt(var)

### 将收益标准化 #####
data2['standard_return'] = data2['return']/data2['sqr_var']
arr_st_return = np.array(data2['standard_return'])
tsplot(arr_st_return, lags=30)

############################################################
####### question four ######
############################################################
data3           = pd.read_excel('question_2.xlsx')

####  return  ####
data3['return'] = np.log(data3.Close/data3['Close'].shift(1))
data3           = data3.dropna()
data3           = data3.reset_index(drop = True)
var             = np.array(data3['s2t']) 
data3['sqr_var'] = np.sqrt(var)

### 将收益标准化 #####
data3['standard_return'] = data3['return']/data3['sqr_var']
arr_st_return = np.array(data3['standard_return'])

############ log likelihood for d‘value ############

def likfunc_td(d,st_return):
    """ 
    inputs: d  - t(d) degrees of freedom
            zs - standardarized returns
    output: logliksum
    
    """
    loglike = []
    zs = st_return
    for i in range(len(st_return)):
       tmp=-scipy.special.gammaln((d+1)/2)+scipy.special.gammaln(d/2)+np.log(np.pi)/2+np.log(d-2)/2+0.5*(1+d)*np.log(1+zs[i]**2/(d-2))
       loglike.append(tmp)       
    return np.array(loglike).sum()

#####  使用极大似然函数估计d值  #####
params_MLE2 = optimize.fmin(likfunc_td,10, args=(arr_st_return,), ftol = 0.00000001)
d  = params_MLE2[0]

#############  学生t-分布的区间点  #############
### way one ##

samp_pct_y       =   data3['standard_return']
fig = sm.qqplot(samp_pct_y, stats.t,  distargs=(d,),fit=True, line='45')


#########################################################################################################
######### question_five #######
########################################
############# ****** Hill Estimator ********** ##################

def EVT_var(data5_1):
    ####  return  ####
    data5_2 = data5_1
    data5_2['return'] = np.log(data5_2.Close/data5_2['Close'].shift(1))
    data5_2           = data5_2.dropna()
    data5_2           = data5_2.reset_index(drop = True)
    
    #### standard the return
    var                = np.array(data5_2['s2t']) 
    data5_2['sqr_var'] = np.sqrt(var)
    data5_2['standard_return'] = data5_2['return']/data5_2['sqr_var']
    #arr_st_return = np.array(data5_2['standard_return'])
    data5              = data5_2.sort_values(by=['standard_return'])
    list_need          =      list(data5.index)
    data5['rank']      =      Series(list(range(2513)),index=list_need)
    data5['rank']      =      data5['rank'].map(lambda x: x+1)
    #####  Choosing the Threshold, u
    thre_u             =      data5.iloc[50,5]
    data5['abs_st_return'] =  data5['standard_return'].abs()
    
    ###### 选出损失最大的50  ######
    dt_5     =   data5[0:50]
    
    ######  Estimating the Tail Index Parameter, ξ( ksi)
    dt_5['ln_u'] = np.log(dt_5['abs_st_return']/abs(thre_u))
    ksi          = (dt_5['ln_u']/float(50)).sum()
    #est_c        = float(50)/len(data5)*abs(thre_u)**(1/float(ksi))
    
    data5_2['var_evt']       = data5_2['sqr_var']*abs(thre_u)*(float(0.01)/(float(50)/len(data5)))**((-1)*ksi)

    return data5_2

data5_1           = pd.read_excel('question_2.xlsx')
data5_2 = EVT_var(data5_1)

def CF_var(data5_1):
    ####  return  ####
    data5_2 = data5_1
    data5_2['return'] = np.log(data5_2.Close/data5_2['Close'].shift(1))
    data5_2           = data5_2.dropna()
    data5_2           = data5_2.reset_index(drop = True)
    
    #### standard the return
    var                = np.array(data5_2['s2t']) 
    data5_2['sqr_var'] = np.sqrt(var)
    data5_2['standard_return'] = data5_2['return']/data5_2['sqr_var']
    arr_st_return = np.array(data5_2['standard_return'])
    ##### 由于精确度不一样，计算结果的偏度和峰度会有差异 #####

    sta = scs.describe(arr_st_return)
    stew = sta[4]
    kurtosis = sta[5]
    
    #### 标准累计分布函数的反函数
    phi_inverse     = norm.isf(0.99, loc=0, scale=1)
    CF_inverse      = phi_inverse+stew/float(6)*(phi_inverse**2-1)+kurtosis/float(24)*(phi_inverse**3-3*phi_inverse)-stew**2/float(36)*(2*phi_inverse**3-5*phi_inverse)
    
    data5_2['CF_VaR']   = data5_2['sqr_var']*CF_inverse*(-1)
    return data5_2

data5_1           = pd.read_excel('question_2.xlsx')
data5_2 = CF_var(data5_1)


def Norm_T_var(data5_1):
    ####  return  ####
    data5_2 = data5_1
    data5_2['return'] = np.log(data5_2.Close/data5_2['Close'].shift(1))
    data5_2           = data5_2.dropna()
    data5_2           = data5_2.reset_index(drop = True)
    var                = np.array(data5_2['s2t']) 
    data5_2['sqr_var'] = np.sqrt(var)
    #### 标准累计分布函数的反函数
    phi_inverse     = norm.isf(0.99, loc=0, scale=1)
    data5_2['norm_VaR'] = phi_inverse*data5_2['sqr_var']*(-1)
    inverse_t = st.t.ppf(0.01,12.26) 
    data5_2['td_VaR'] = (-1)*data5_2['sqr_var']*np.sqrt((d-2)/float(d))*inverse_t
    return data5_2

    
data5_1           = pd.read_excel('question_2.xlsx')
Norm_T_var(data5_1)


#####################################################################################
################ question six ###############
##################################################
data6                  = pd.read_csv('question_4.xlsx')

len(data6)
#####  Choosing the Threshold, u
thre_u   =   data6.iloc[50,2]
data6['abs_st_return'] = data6['standard_return'].abs()

#####  选出损失最大的50   ######
dt_6     =   data6[0:50]
dt_6['ln_u'] = np.log(dt_6['abs_st_return']/abs(thre_u))
ksi          = (dt_6['ln_u']/float(50)).sum()
ax_x     = []
for i in range(50):
    ii = i+1
    ax_x.append(thre_u*((ii-0.5)/float(50))**((-1)*ksi))
dt_6.loc[:,'evt_left_tail'] = np.array(ax_x)

######  作图 plot  ####

y_list      = list(data6['standard_return'])
plt.scatter(x=ax_x, y=y_list[0:50], color='blue')
plt.xlim((-8, 8))
plt.ylim((-8, 8))
plt.grid()
plt.plot([-8, 8], [-8, 8], linewidth=2)
#####################################################################################
################ question seven ###############
##################################################
### load the need data ###
data_7 = pd.read_excel('question_7.xlsx')

p      = 1
window = 251
data_7['sigma2'] = np.nan
return_need = list(data_7['Return'][1:252])
var        = np.var(return_need)
data_7.loc[252,'sigma2'] = var

def RM_var(data_7):
    for i in range(253,(data_7.shape[0])):
        data_7.loc[i,'sigma2'] = data_7.loc[i-1,'sigma2']*0.94 + 0.06* data_7.loc[i-1,'Return']**2
        data_7['VAR_RM'] = -data_7['sigma2']**0.5 * norm(0,1).ppf(0.01)
    return data_7
data_7 = RM_var(data_7)

def Hist_var(data_7):
    for i in range(window + 1,data_7.shape[0]):
        datause = data_7.loc[i - window:i - 1,'Return']
        data_7.loc[i,'Hist_VaR'] = - np.percentile(datause,p)
    return data_7
data_7 = Hist_var(data_7)
        
def Fhist_var(data_7):
    for i in range(window + 1,data_7.shape[0]):
        datause = data_7.loc[i - window:i - 1,'Standardized Return']
        data_7.loc[252:504,'fhist_VaR'] = (-1)*np.sqrt(data_7.loc[252:504,'NGARCH s2t'])*np.percentile(datause,p)
    return data_7
data_7 =Fhist_var(data_7)

data = data5_2.loc[2260:2513,'td_VaR'].copy()
data_7.loc[252:504,'td_VaR']   = data


        



    







    


