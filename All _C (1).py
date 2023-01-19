import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing

def read_csv():
    """
    Creates 3 dataframes and return them. The dataset related to world bank climate change datasets. 
    """
    df1=pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_4770565.csv',skiprows=4).fillna(0)

    df2=pd.read_csv('API_EN.ATM.CO2E.PP.GD_DS2_en_csv_v2_4771886.csv',skiprows=4).fillna(0)

    df3=pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv',skiprows=4).fillna(0)
    return df1,df2,df3

def get_lists():
    """
    Normalization and transform data.
    Return three lists.
    li_CO2P -> Used for  CO2 per PPP data in 2019 year after normalize data
    li_CO2C -> Used for  CO2 per Capta data in 2019 year after normalize data
    li_GDP  -> Used for  GDP per Capta data in 2019 year after normalize data
    """
    li_CO2P=preprocessing.normalize([np.array(list(df1["2019"]))])[0]
    li_CO2C=preprocessing.normalize([np.array(list(df2["2019"]))])[0]
    li_GDP=preprocessing.normalize([np.array(list(df3["2019"]))])[0]
    return li_CO2P,li_CO2C,li_GDP

def clustering_1990():
    """
    Use the world bank data of selected countries' CO2 generation PPP,
    CO2 generation per capita and GDP per capita in 1990. And clustered the 3d plot using kmeans with 3 clusters.
    """
    li_CO2P1=preprocessing.normalize([np.array(list(df1["1990"]))])[0]
    li_CO2C1=preprocessing.normalize([np.array(list(df2["1990"]))])[0]
    li_GDP1=preprocessing.normalize([np.array(list(df3["1990"]))])[0]
    ncluster=3
    kmeans=cluster.KMeans(n_clusters=ncluster)
    XYZ1=[]
    for r in range(len(li_GDP)):
        po=[li_CO2P1[r],li_CO2C1[r],li_GDP1[r]]
        XYZ1.append(po)
    KM1=kmeans.fit(XYZ1)
    KM1.cluster_centers_
    Clu_cen_X1=[]
    Clu_cen_Y1=[]
    Clu_cen_Z1=[]
    for r in range(len(KM1.cluster_centers_)):
        Clu_cen_X1.append(KM1.cluster_centers_[r][0])
        Clu_cen_Y1.append(KM1.cluster_centers_[r][1])
        Clu_cen_Z1.append(KM1.cluster_centers_[r][2])

    colors=["green","blue","red"]
    label1=KM1.labels_
    label_c1=[]
    X1_0=[]
    X1_1=[]
    X1_2=[]
    Y1_0=[]
    Y1_1=[]
    Y1_2=[]
    Z1_0=[]
    Z1_1=[]
    Z1_2=[]
    for r in range(len(label1)):
        label_c1.append(colors[label1[r]])
        if KM1.labels_[r]==0:
            X1_0.append(li_CO2P1[r])
            Y1_0.append(li_CO2C1[r])
            Z1_0.append(li_GDP1[r])
        elif KM1.labels_[r]==1:
            X1_1.append(li_CO2P1[r])
            Y1_1.append(li_CO2C1[r])
            Z1_1.append(li_GDP1[r])
        elif KM1.labels_[r]==2:
            X1_2.append(li_CO2P1[r])
            Y1_2.append(li_CO2C1[r])
            Z1_2.append(li_GDP1[r])
    fig=plt.figure(figsize=(10,10))
    ax=plt.axes(projection="3d")
    ax.scatter(X1_0,Y1_0,Z1_0,c="green",label="Cluster 1")
    ax.scatter(X1_1,Y1_1,Z1_1,c="blue",label="Cluster 2")
    ax.scatter(X1_2,Y1_2,Z1_2,c="red",label="Cluster 3")
    ax.scatter(Clu_cen_X1,Clu_cen_Y1,Clu_cen_Z1,c=colors,marker="x",s=300,label="Cluster Centers")
    ax.legend()
    ax.set_xlabel("CO2/PPP")
    ax.set_ylabel("CO2 per capita")
    ax.set_zlabel("GDP per capita")
    plt.title("K-Means Clustering for data in 1990")
    plt.show()


def clustering_2019():
    """
    For the KMeans clustering plot for 2019, Used the world bank data of selected countries'
    CO2 generation PPP, CO2 generation
    per capita and GDP per capita in 2019. And clustered the 3d plot using kmeans with 3 clusters.
    """
    ncluster=3
    kmeans=cluster.KMeans(n_clusters=ncluster)
    XYZ=[]
    for r in range(len(li_GDP)):
        po=[li_CO2P[r],li_CO2C[r],li_GDP[r]]
        XYZ.append(po)
    KM=kmeans.fit(XYZ)
    KM.cluster_centers_
    Clu_cen_X=[]
    Clu_cen_Y=[]
    Clu_cen_Z=[]
    for r in range(len(KM.cluster_centers_)):
        Clu_cen_X.append(KM.cluster_centers_[r][0])
        Clu_cen_Y.append(KM.cluster_centers_[r][1])
        Clu_cen_Z.append(KM.cluster_centers_[r][2])

    colors=["green","blue","red"]
    label=KM.labels_
    label_c=[]
    X_0=[]
    X_1=[]
    X_2=[]
    Y_0=[]
    Y_1=[]
    Y_2=[]
    Z_0=[]
    Z_1=[]
    Z_2=[]
    for r in range(len(label)):
        label_c.append(colors[label[r]])
        if KM.labels_[r]==0:
            X_0.append(li_CO2P[r])
            Y_0.append(li_CO2C[r])
            Z_0.append(li_GDP[r])
        elif KM.labels_[r]==1:
            X_1.append(li_CO2P[r])
            Y_1.append(li_CO2C[r])
            Z_1.append(li_GDP[r])
        elif KM.labels_[r]==2:
            X_2.append(li_CO2P[r])
            Y_2.append(li_CO2C[r])
            Z_2.append(li_GDP[r])
    
    fig=plt.figure(figsize=(10,10))
    ax=plt.axes(projection="3d")
    ax.scatter(X_0,Y_0,Z_0,c="green",label="Cluster 1")
    ax.scatter(X_1,Y_1,Z_1,c="blue",label="Cluster 2")
    ax.scatter(X_2,Y_2,Z_2,c="red",label="Cluster 3")
    ax.scatter(Clu_cen_X,Clu_cen_Y,Clu_cen_Z,c=colors,marker="x",s=300,label="Cluster Centers")
    ax.legend()
    ax.set_xlabel("CO2/PPP")
    ax.set_ylabel("CO2 per capita")
    ax.set_zlabel("GDP per capita")
    plt.title("K-Means Clustering for data in 2019")
    plt.show()
    
def initialize():
    """
    Function for initialize data.
    Return dataframe, year, argentina/Spain/Switzerland GDP throughout the years
    """
    df=df3
    df3.iloc[37][1:-2]
    year=np.array(list(range(1960,2021)))
    ArgGDP=np.array(list(df3.iloc[9][4:-2]))
    SpnGDP=list(df3.iloc[70][4:-2])
    SwisGDP=list(df3.iloc[37][4:-2])
    return df,year,ArgGDP,SpnGDP,SwisGDP



def linfunc(x,a,b):
    """ Function for fitting
        x: independent variable
        a, b: parameters to be fitted
    """
    y=a*x+b
    return y

def err_ranges(x, func, param, sigma): #function with 3 imputs as x func, param, sigma
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   

def initialize_param_1():
    """
    Function for initialize parameters for fitting data in following plot.
    plot - GDP of Argentina throughout years
    """
    a1,b1=200,-392500
    x=year
    y=linfunc(x,a1,b1)
    errors=np.ones_like(x)
    return a1,b1,x,y,errors
    


def gdpArgentinaPlot():
    """
    Function for plot creation.
    Show the fitting data for GDP in Argentina Country through out the years
    """
    a1,b1,x,y,errors=initialize_param_1()
    plt.scatter(year,ArgGDP)
    plt.title("GDP of Argentina throughout years")
    plt.ylabel("GDP")
    plt.xlabel("Year")
    plt.errorbar(x,y,errors,color="black",linestyle="none")
    plt.plot(x,y,c="red",label="Fitted line")
    plt.legend()
    plt.show()

    param,pcovar=opt.curve_fit(linfunc,x,ArgGDP)
    print("Parameters: ",param)
    print()
    print("Covariance-matrix",pcovar)
    sigma=np.sqrt(np.diag(pcovar))
    print()
    print(f"a={param[0]:5.3f}+/-{sigma[0]:5.3f}")
    print(f"b={param[1]:5.3f}+/-{sigma[1]:5.3f}")
    
    X=np.array(list(range(2021,2041)))
    func=linfunc
    LL,UL=err_ranges(X, func, param, sigma)

    print("Predictions for 20 years starting from 2021")
    print()
    Y=[]
    for r in range(20):
        va=linfunc(2021+r,a1,b1)
        Y.append(va)
        st=str(2021+r)+" - "+str(va)
        print(st)
        print("Confidence interval Upper Limit is "+str(UL[r]))
        print("Confidence interval Lower Limit is "+str(LL[r]))
        print()
    Y=np.array(Y)


def initialize_param_2():
    """
    Function for initialize parameters for fitting data in following plot.
    plot - GDP of Switzerland throughout years
    """
    a2,b2=1670,-3282500
    x=year
    y=linfunc(x,a2,b2)
    errors=np.ones_like(x)
    return a2,b2,x,y,errors

def gdpSwitzerlandPlot():
    """
    Function for plot creation.
    Show the fitting data for GDP in Switzerland Country through out the years
    """
    a2,b2,x,y,errors = initialize_param_2()
    plt.scatter(year,SwisGDP)
    plt.title("GDP of Switzerland throughout years")
    plt.ylabel("GDP")
    plt.xlabel("Year")
    plt.errorbar(x,y,errors,color="black",linestyle="none")
    plt.plot(x,y,c="red",label="Fitted line")
    plt.legend()
    plt.show()

    param,pcovar=opt.curve_fit(linfunc,x,SwisGDP)
    print("Parameters: ",param)
    print()
    print("Covariance-matrix",pcovar)
    sigma=np.sqrt(np.diag(pcovar))
    print()
    print(f"a={param[0]:5.3f}+/-{sigma[0]:5.3f}")
    print(f"b={param[1]:5.3f}+/-{sigma[1]:5.3f}")
    
    X=np.array(list(range(2021,2041)))
    func=linfunc
    LL,UL=err_ranges(X, func, param, sigma)

    print("Predictions for 20 years starting from 2021")
    print()
    Y=[]
    for r in range(20):
        va=linfunc(2021+r,a2,b2)
        Y.append(va)
        st=str(2021+r)+" - "+str(va)
        print(st)
        print("Confidence interval Upper Limit is "+str(UL[r]))
        print("Confidence interval Lower Limit is "+str(LL[r]))
        print()
    Y=np.array(Y)
    plt.scatter(X,Y)
    plt.title("GDP of Argentina predictions throughout years")
    plt.ylabel("GDP Predictions")
    plt.xlabel("Year")
    plt.show()
    ER=UL-LL
    plt.scatter(X,ER)
    plt.title("GDP of Argentina predictions errors and confidence intervals ranges throughout years")
    plt.ylabel("GDP error ranges ")
    plt.xlabel("Year")
    plt.show()


def initialize_param_3():
    """
    Function for initialize parameters for fitting data in following plot.
    plot - GDP of Spain throughout years
    """
    a3,b3=583.33,-1147500
    x=year
    y=linfunc(x,a3,b3)
    errors=np.ones_like(x)
    return a3,b3,x,y,errors

def gdpSpainPlot():
    """
    Function for plot creation.
    Show the fitting data for GDP in Spain Country through out the years
    """
    a3,b3,x,y,errors = initialize_param_2()
    plt.scatter(year,SpnGDP)
    plt.title("GDP of Spain throughout years")
    plt.ylabel("GDP")
    plt.xlabel("Year")
    plt.errorbar(x,y,errors,color="black",linestyle="none")
    plt.plot(x,y,c="red",label="Fitted line")
    plt.legend()
    plt.show()

    param,pcovar=opt.curve_fit(linfunc,x,SpnGDP)
    print("Parameters: ",param)
    print()
    print("Covariance-matrix",pcovar)
    sigma=np.sqrt(np.diag(pcovar))
    print()
    print(f"a={param[0]:5.3f}+/-{sigma[0]:5.3f}")
    print(f"b={param[1]:5.3f}+/-{sigma[1]:5.3f}")
    
    X=np.array(list(range(2021,2041)))
    func=linfunc
    LL,UL=err_ranges(X, func, param, sigma)

    print("Predictions for 20 years starting from 2021")
    print()
    Y=[]
    for r in range(20):
        va=linfunc(2021+r,a3,b3)
        Y.append(va)
        st=str(2021+r)+" - "+str(va)
        print(st)
        print("Confidence interval Upper Limit is "+str(UL[r]))
        print("Confidence interval Lower Limit is "+str(LL[r]))
        print()
    Y=np.array(Y)

#Main

df1,df2,df3 = read_csv()
li_CO2P,li_CO2C,li_GDP = get_lists()
clustering_1990()
clustering_2019()
df,year,ArgGDP,SpnGDP,SwisGDP = initialize()
gdpArgentinaPlot()
gdpSwitzerlandPlot()
gdpSpainPlot()
