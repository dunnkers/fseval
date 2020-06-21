'''

This package implements the statistical tools for quantifying the stability of feature selection as given in [1].
It includes 5 functions that provide:
    - the stability estimate of a feature selection procedure given its outputs;
    - the variance of the stability estimate;
    - a (1-alpha)- approximate confidence intervals for the population stability;
    - a null hypothesis test allowing to compare the population stability of a feature selection procedure to 
      a given value.
    - a null hypothesis test allowing to compare the population stabilities of two feature selection procedures.

[1] On the Stability of Feature Selection. Sarah Nogueira, Konstantinos Sechidis, Gavin Brown. 
    Journal of Machine Learning Reasearch (JMLR). 2017.

You can find a full demo using this package at:
http://htmlpreview.github.io/?https://github.com/nogueirs/JMLR2017/blob/master/python/stabilityDemo.html

NB: This package requires the installation of the packages: numpy, scipy and math

'''

import numpy as np
from scipy.stats import norm
import math


def getStability(Z):
    ''' 
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].
    
    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
           means the f^th feature has been selected and a 0 means it has not been selected.
           
    OUTPUT: The stability of the feature selection procedure
    '''
    Z=checkInputType(Z)
    M,d=Z.shape
    hatPF=np.mean(Z,axis=0)
    kbar=np.sum(hatPF)
    denom=(kbar/d)*(1-kbar/d)
    return 1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom

def getVarianceofStability(Z):
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate and its variance as given in [1].
    
    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
           means the f^th feature has been selected and a 0 means it has not been selected.
           
    OUTPUT: A dictionnary where the key 'stability' provides the corresponding stability value #
            and where the key 'variance' provides the variance of the stability estimate
    '''
    Z=checkInputType(Z) # check the input Z is of the right type
    M,d=Z.shape # M is the number of feature sets and d the total number of features
    hatPF=np.mean(Z,axis=0) # hatPF is a numpy.array with the frequency of selection of each feature
    kbar=np.sum(hatPF) # kbar is the average number of selected features over the M feature sets
    k=np.sum(Z,axis=1) # k is a numpy.array with the number of features selected on each one of the M feature sets
    denom=(kbar/d)*(1-kbar/d) 
    stab=1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom # the stability estimate
    phi=np.zeros(M)
    for i in range(M):
        phi[i]=(1/denom)*(np.mean(np.multiply(Z[i,],hatPF))-(k[i]*kbar)/d**2+(stab/2)*((2*k[i]*kbar)/d**2-k[i]/d-kbar/d+1))
    phiAv=np.mean(phi)
    variance=(4/M**2)*np.sum(np.power(phi-phiAv,2)) # the variance of the stability estimate as given in [1]
    return {'stability':stab,'variance':variance}

def confidenceIntervals(Z,alpha=0.05,res={}):
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function provides the stability estimate and the lower and upper bounds of the (1-alpha)- approximate confidence 
    interval as given by Corollary 9 in [1]
    
    INPUTS: - A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise). 
              Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
              means the f^th feature has been selected and a 0 means it has not been selected.
            - alpha is an optional argument corresponding to the level of significance for the confidence interval 
              (default is 0.05), e.g. alpha=0.05 give the lower and upper bound of for a (1-alpha)=95% confidence interval.
            - In case you already computed the stability estimate of Z using the function getVarianceofStability(Z), 
              you can provide theresult (a dictionnary) as an optional argument to this function for faster computation.
           
    OUTPUT: - A dictionnary where the key 'stability' provides the corresponding stability value, where:
                  - the key 'variance' provides the variance of the stability estimate;
                  - the keys 'lower' and 'upper' respectively give the lower and upper bounds 
                    of the (1-alpha)-confidence interval.
    '''
    Z=checkInputType(Z) # check the input Z is of the right type
    ## we check if values of alpha between ) and 1
    if alpha>=1 or alpha<=0:
        raise ValueError('The level of significance alpha should be a value >0 and <1')
    if len(res)==0: 
        res=getVarianceofStability(Z) # get a dictionnary with the stability estimate and its variance
    lower=res['stability']-norm.ppf(1-alpha/2)*math.sqrt(res['variance']) # lower bound of the confidence interval at a level alpha
    upper=res['stability']+norm.ppf(1-alpha/2)*math.sqrt(res['variance']) # upper bound of the confidence interval 
    return {'stability':res['stability'],'lower':lower,'upper':upper}

## this tests whether the true stability is equal to a given value stab0
def hypothesisTestV(Z,stab0,alpha=0.05):
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test in [1] that test whether the population stability is greater 
    than a given value stab0.
    
    INPUTS:- A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise).
             Each row of the binary matrix represents a feature set, where a 1 at the f^th position
             means the f^th feature has been selected and a 0 means it has not been selected. 
           - stab0 is the value we want to compare the stability of the feature selection to.
           - alpha is an optional argument corresponding to the level of significance of the null hypothesis test 
             (default is 0.05).
           
    OUTPUT: A dictionnary with:
            - a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            - a float for the key 'V' giving the value of the test statistic 
            - a float giving for the key 'p-value' giving the p-value of the hypothesis test
    '''
    Z=checkInputType(Z) # check the input Z is of the right type
    res=getVarianceofStability(Z)
    V=(res['stability']-stab0)/math.sqrt(res['variance'])
    zCrit=norm.ppf(1-alpha)
    if V>=zCrit: reject=True
    else: reject=False
    pValue=1-norm.cdf(V)
    return {'reject':reject,'V':V,'p-value':pValue}

# this tests the equality of the stability of two algorithms
def hypothesisTestT(Z1,Z2,alpha=0.05):
    '''
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function implements the null hypothesis test of Theorem 10 in [1] that test whether 
    two population stabilities are identical.
    
    INPUTS:- Two BINARY matrices Z1 and Z2 (given as lists or as numpy.ndarray objects of size M*d).
             Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
             means the f^th feature has been selected and a 0 means it has not been selected. 
           - alpha is an optional argument corresponding to the level of significance of the null 
             hypothesis test (default is 0.05)
           
    OUTPUT: A dictionnary with:
            - a boolean value for key 'reject' equal to True if the null hypothesis is rejected and to False otherwise
            - a float for the key 'T' giving the value of the test statistic 
            - a float giving for the key 'p-value' giving the p-value of the hypothesis test
    '''
    Z1=checkInputType(Z1) # check the input Z1 is of the right type
    Z2=checkInputType(Z2) # check the input Z2 is of the right type
    res1=getVarianceofStability(Z1)
    res2=getVarianceofStability(Z2)
    stab1=res1['stability']
    stab2=res2['stability']
    var1=res1['variance']
    var2=res2['variance']
    T=(stab2-stab1)/math.sqrt(var1+var2)
    zCrit=norm.ppf(1-alpha/2) 
    ## the cumulative inverse of the gaussian at 1-alpha/2
    if(abs(T)>=zCrit):
        reject=True
        #print('Reject H0: the two algorithms have different population stabilities')
    else:
        reject=False
        #print('Do not reject H0')
    pValue=2*(1-norm.cdf(abs(T)))
    return {'reject':reject,'T':T,'p-value':pValue}

def checkInputType(Z):
    ''' This function checks that Z is of the rigt type and dimension.
        It raises an exception if not.
        OUTPUT: The input Z as a numpy.ndarray
    '''
    ### We check that Z is a list or a numpy.array
    if isinstance(Z,list):
        Z=np.asarray(Z)
    elif not isinstance(Z,np.ndarray):
        raise ValueError('The input matrix Z should be of type list or numpy.ndarray')
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim!=2:
        raise ValueError('The input matrix Z should be of dimension 2')
    return Z