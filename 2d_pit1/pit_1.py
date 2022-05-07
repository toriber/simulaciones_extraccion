import numpy as np
import pandas as pd

#Generar los espacios
X_cord,Y_cord=np.mgrid[-5:5:0.1,0:5:0.1]
xy = np.vstack((X_cord.flatten(), Y_cord.flatten())).T

#tomaremos el epigrafo de 2x^2 como Gamma, y el pit será el epigrafo de |x|

#generar el pit
def get_pit_abs(XY):
    pit=np.array(list())
    for point in XY:
        if np.abs(point[0]) <= point[1]:
            pit=np.append(pit,point)
    return np.reshape(pit,(-1,2))

#intentaremos con un g el cual sea negativo en 0, una parte positiva, y el techo negativo
def g_func(point):
    return -(point[1]-2.7)**2+3

def gamma_sq(x0,pit):
    Gamma=np.array(list())
    for point in pit:
        if (1/3)*(point[0]-x0[0])**2 +x0[1] <=point[1]:
            Gamma=np.append(Gamma,point)
    return np.reshape(Gamma,(-1,2))


def get_XY(pit,g_func):
    X=np.array(list())
    Y=np.array(list())
    for point in pit:
        if g_func(point)>0:
            X=np.append(X,point)
        if g_func(point)<0:
            Y=np.append(Y,point)
    return np.reshape(X,(-1,2)),np.reshape(Y,(-1,2))

def func_c(X,Y,Gamma):
    c=np.full((X.shape[0],Y.shape[0]),10**9)
    i=0
    for x in X:
        j=0
        gamma=Gamma(x).tolist()
        for y in Y:
            y_list=y.tolist()
            if y_list in gamma:
                c[i,j]=0
            j+=1
        i+=1
        if i % 100==0:
            print("Has been "+str(i)+" iterations")
    return c
        

pit=get_pit_abs(xy)
X_no_alpha,Y_no_w=get_XY(pit,lambda point: g_func(point))
c=func_c(X_no_alpha,Y_no_w,lambda point: gamma_sq(point,pit))
c_new=np.append(c,np.ones((X_no_alpha.shape[0],1)),axis=1)
c_new=np.append(c_new,np.zeros((Y_no_w.shape[0]+1,1)).T,axis=0)
mu=np.array(list())
nu=np.array(list())
for x in X_no_alpha:
    mu=np.append(mu,g_func(x)*0.1)
for y in Y_no_w:
    nu=np.append(nu,abs(g_func(y))*0.1)

alpha=sum(nu)
w=sum(mu)
mu=np.append(mu,alpha)
nu=np.append(nu,w)
mu=mu*10
nu=nu*10
X=np.append(X_no_alpha,["alpha","alpha"])
X=np.reshape(X,(-1,2))
Y=np.append(Y_no_w,["w","w"])
Y=np.reshape(Y,(-1,2))
p=list()
for i in range(X.shape[0]):
    p.append("p_"+str(i))
q=list()
for j in range(Y.shape[0]):
    q.append("q_"+str(j))

############## SOLVING THE PROBLEM ##########
from pulp import *

prob=LpProblem("Pit_Problem",LpMaximize)

p_var= LpVariable.dicts("p",p,0,1,cat="Integer")

q_var=LpVariable.dicts("q",q,0,1,cat="Integer")
c_df=pd.DataFrame(c_new)
c_df.columns=q
c_df=c_df.T
c_df.columns=p
mu_dict=dict(zip(p,mu))

nu_dict=dict(zip(q,nu))
prob+= lpSum([mu_dict[i]*p_var[i] for i in p] + [-nu_dict[i]*q_var[i] for i in q])
for i in p:
    for j in q:
        prob+= p_var[i]-q_var[j] <=float(c_df[i][j])

solver=PULP_CBC_CMD(msg=True)
print(prob)
result=prob.solve(solver)       
variables=prob.variables()
p_opt=np.zeros(X.shape[0])
q_opt=np.zeros(Y.shape[0])
for v in prob.variables():
    if "p" in v.name:
        p_opt[int(re.findall(r'\d+',v.name)[0])]=v.varValue
    else:
        q_opt[int(re.findall(r'\d+',v.name)[0])]=v.varValue

        
np.savetxt("p_opt_pit1.csv", p_opt, delimiter=",")
np.savetxt("q_opt_pit1.csv", q_opt, delimiter=",")