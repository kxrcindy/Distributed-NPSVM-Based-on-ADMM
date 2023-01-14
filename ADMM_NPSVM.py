# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:11:15 2021

@author: 26878
"""

import numpy as np
import pandas as pd

def ADMM_NPSVM(data,c,epsilon,rho):
    Y=data[:,0]
    X=np.delete(data,0,1)
    pos=X[Y==1]
    neg=X[Y==(0)]
    p=pos.shape[0]
    q=neg.shape[0]
    w1,b1=NPSVM_RES(pos,neg,c,rho,epsilon,1,100)
    w2,b2=NPSVM_RES(neg,pos,c,rho,epsilon,-1,100)
    sv1=0
    sv2=0
    R1=np.dot(pos,w1)+b1
    R2=np.dot(pos,w2)+b2
    R3=np.dot(neg,w1)+b1
    R4=np.dot(neg,w2)+b2
    for i in range(0,p):
        if R1[i]>=epsilon and R2[i]<=1:
            sv1=sv1+1
    for j in range(0,q):
        if R4[j]>=epsilon and R3[j]>=-1:
            sv2=sv2+1
    acc=(p+q-sv1-sv2)/(p+q)
    return acc



def NPSVM_RES(thisdata,otherdata,c,rho,epsilon,indicate,MAX_ITER=100):
    ABSTOL   = 1e-4
    RELTOL   = 1e-2
    rval=np.zeros((MAX_ITER,1))
    sval=np.zeros((MAX_ITER,1))
    eps_pri=np.zeros((MAX_ITER,1))
    eps_dual=np.zeros((MAX_ITER,1))
    m1,n1=thisdata.shape
    m2,n2=otherdata.shape
    thisdata=np.hstack((thisdata,np.ones((m1,1))))
    otherdata=np.hstack((otherdata,np.ones((m2,1))))
    #indicator=1/-1
    if (indicate==(-1)):
        otherdata=-otherdata
        T=np.vstack((-thisdata,thisdata,-otherdata))
        np.savetxt("C:/Users/26878/Desktop/negative.csv",T,delimiter=",")
    else:
        T=np.vstack((-thisdata,thisdata,-otherdata))
        np.savetxt("C:/Users/26878/Desktop/positive.csv",T,delimiter=",")   
    C=c*np.ones((2*m1+m2,1))
    z1=-epsilon*np.ones((m1,1))
    z2=np.ones((m2,1))
    z=np.vstack((z1,z1,z2))
    L=np.zeros((2*m1+m2,1))
    U=np.zeros((2*m1+m2,1))
    A=np.eye(n1+1)+rho*(np.dot(T.T,T))
    for i in range(0,MAX_ITER):
        #updata P
        b=-rho*np.dot(T.T,L-z+U)
        P=CG(A,b,e=1e-5,maxiters=1000)
        #update L
        Lold=L
        Ltemp=z-np.dot(T,P)-U
        t=C/rho
        for j in range(0,2*m1+m2):
            if Ltemp[j]>t[j]:
                L[j]=Ltemp[j]-t[j]
            elif Ltemp[j]<0:
                L[j]=Ltemp[j]
            else:
                L[j]=0
        #update U
        U=U+np.dot(T,P)-z
        # check
        r=np.dot(T,P)+L-z
        Lup=L-Lold
        s=rho*np.dot(T.T,Lup)
        rval[i]=np.dot(r.T,r)
        sval[i]=np.dot(s.T,s)
        TtU=np.dot(T.T,U)
        eps_pri[i] = np.sqrt(2*m1+m2+2)*ABSTOL + RELTOL*max(np.linalg.norm(np.dot(T,P)), np.linalg.norm(L), np.linalg.norm(z))
        eps_dual[i]= np.sqrt(2*(2*m1+m2))*ABSTOL + RELTOL*np.linalg.norm(rho*TtU)
        if  rval[i] < eps_pri[i] and sval[i] < eps_dual[i]:
            break
    w=P[:n1]
    beta=P[-1]
    return w,beta


def CG(A,b,e=1e-5,maxiters=1000):
    m=b.shape[0]
    x=np.zeros((m,1))
    r=b-np.dot(A,x)
    p=r
    for i in range(0,maxiters):
        Ap=np.dot(A,p)
        alpha=np.dot(r.T,r)/np.dot(p.T,Ap)
        #alpha=alpha[0][0]
        x=x+alpha*p
        rnew=r-alpha*Ap
        if np.dot(r.T,r)<=e:
            break
        beta=np.dot(rnew.T,rnew)/np.dot(r.T,r)
        #beta=beta[0][0]
        p=rnew+beta*p
        r=rnew
    return x

np.random.seed(5)
b=np.random.randint(0,10,(10,1))
A=np.random.randint(0,100,(10,10))
A=np.dot(A.T,A)
ans1=np.dot(np.linalg.inv(A),b)
ans2=CG(A,b,e=1e-5)
print(ans1)
print(ans2)
P=np.random.randint(0,100,(150,10))
N=np.random.randint(0,100,(100,10))
res,bias=NPSVM_RES(P,N,1,1,1e-5,1,MAX_ITER=100)
data=pd.read_csv("C:/Users/26878/Desktop/final_data.csv")
data=data.head(3000)
data=np.array(data)
a=ADMM_NPSVM(data,1,0.1,1)