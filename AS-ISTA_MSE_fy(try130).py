#ステップサイズを入力として各層のMSEと目的関数の値のグラフを出力する

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime as dt
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

m = 75 # 観測ベクトル次元
n = 150 # 元信号ベクトル次元
snr = 10
k = 12
p = k/n # 元信号の非ゼロ元の生起確率
sigma = np.sqrt(p/(10**(snr/10)))
mbs1   = 100# ミニバッチサイズ（xの数）
mbs2 = 100  #Aの数
lam = 10  #正則化係数
max_itr = 100 #反復回数
device = torch.device('cuda') # 'cpu' or 'cuda' 


#l0ノルムの近接写像
def prox_L0(x, tau):
  epsilon = 1e-10
  th = np.sqrt(2 * tau + epsilon)
  return np.where(np.abs(x) < th, 0, x)

#l(1/2)ノルムの近接写像(近似)
def prox_L12(x, tau):
    sigma = 0.1
    epsilon = 1e-10
    tau_pos = tau + epsilon
    th2 = 3 / 2 * (tau_pos ** (2/3))
    th1 = th2 * (1 - sigma)
        
    prox1 = 2 / (3 * sigma) * x + np.sign(x) * (1 - 1 / sigma) * (tau_pos ** (2/3))
    prox2 = 2 / 3 * x * (1 + np.cos(2 / 3 * np.arccos( np.clip(-3 ** (3/2) / 4 * tau_pos * ((np.abs(x)+epsilon) ** (-3/2)), a_min=-1, a_max=1) ) ))
    
    return np.where(np.abs(x) <= th1, 0, prox1) + np.where(np.abs(x) <= th2, 0, prox2 - prox1)

#l1ノルムの近接写像
def prox_L1(x, tau):
  return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

#L0ノルム
def l0_norm(x):
  return np.sum(np.where(np.abs(x)>0,1,0))
#L1/2ノルム
def l12_norm(x):
  return np.sum((np.abs(x)+1e-10)**(1/2))
#L1ノルム
def l1_norm(x):
  return np.sum(np.abs(x))

#ミニバッチ生成関数
def gen_minibatch():
    seq = torch.normal(torch.zeros(mbs1, n), 1.0) # ガウス乱数ベクトルの生成
    support = torch.bernoulli(p * torch.ones(mbs1, n)) # 非ゼロサポートの生成
    return seq * support # 要素ごとの積(アダマール積)になることに注意

def GD_step(x, tau):
        return x + tau * (y - x @ A.T) @ A
def S_step(x, tau):
        return prox_L1(x, lam * tau)        
    
# coef_GD1=np.zeros(max_itr)
# coef_GD2=np.zeros(max_itr)
# coef_ST1=np.zeros(max_itr)
# coef_ST2=np.zeros(max_itr)

#ISTAの関数(MSEを出力)
def ISTA_MSE(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))     
  for i in range(0,itr-1):
    r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))     
  return loss1/mbs1

#ISTAの関数(目的関数の値を出力)
def ISTA_Fy(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  for i in range(itr):
    r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  return loss2/mbs1

def vanilla_ISTA_MSE(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))     
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))
  for i in range(0,itr-1):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))     
  return loss1/mbs1

def vanilla_ISTA_Fy(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/n)*np.sum(np.square(x-s))
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))     
  for i in range(itr):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)    
  return loss2/mbs1

def vanilla_LISTA_MSE(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))     
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))
  for i in range(0,itr-1):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))     
  return loss1/mbs1

def vanilla_LISTA_Fy(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/n)*np.sum(np.square(x-s))
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))     
  for i in range(itr):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)    
  return loss2/mbs1

#ISTAの関数(MSEを出力)
def ISTA_MSE2(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))
  for i in range(0,itr-1):
    #r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    #s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    #r=(beta1[i]/(beta1[i]+beta2[i])) * GD_step(s, alpha[i])+(beta2[i]/(beta1[i]+beta2[i])) * S_step(s, alpha[i])
    #s=(gamma1[i]/(gamma1[i]+gamma2[i])) * GD_step(r, alpha[i])+(gamma2[i]/(gamma1[i]+gamma2[i])) * S_step(r, alpha[i])
    r = (beta1[i]) * GD_step(s, alpha[i]) + (1- (beta1[i])) * S_step(s,alpha[i])
    s = (1- (gamma2[i])) * GD_step(r, alpha[i]) + (gamma2[i]) * S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))
  return loss1/mbs1

#ISTAの関数(目的関数の値を出力)
def ISTA_Fy2(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  for i in range(itr):
    #r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    #s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    #r=(beta1[i]/(beta1[i]+beta2[i])) * GD_step(s, alpha[i])+(beta2[i]/(beta1[i]+beta2[i])) * S_step(s, alpha[i])
    #s=(gamma1[i]/(gamma1[i]+gamma2[i])) * GD_step(r, alpha[i])+(gamma2[i]/(gamma1[i]+gamma2[i])) * S_step(r, alpha[i])
    r = (beta1[i]) * GD_step(s, alpha[i]) + (1- (beta1[i])) * S_step(s,alpha[i])
    s = (1- (gamma2[i])) * GD_step(r, alpha[i]) + (gamma2[i]) * S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  return loss2/mbs1

#ISTAの関数(MSEを出力)
def ISTA_MSE3(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))
  for i in range(0,itr-1):
    #r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    #s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    r=(beta1[i]/(beta1[i]+beta2[i])) * GD_step(s, alpha[i])+(beta2[i]/(beta1[i]+beta2[i])) * S_step(s, alpha[i])
    s=(gamma1[i]/(gamma1[i]+gamma2[i])) * GD_step(r, alpha[i])+(gamma2[i]/(gamma1[i]+gamma2[i])) * S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))
  return loss1/mbs1



#ISTAの関数(目的関数の値を出力)
def ISTA_Fy3(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  for i in range(itr):
    #r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    #s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    r=(beta1[i]/(beta1[i]+beta2[i])) * GD_step(s, alpha[i])+(beta2[i]/(beta1[i]+beta2[i])) * S_step(s, alpha[i])
    s=(gamma1[i]/(gamma1[i]+gamma2[i])) * GD_step(r, alpha[i])+(gamma2[i]/(gamma1[i]+gamma2[i])) * S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  return loss2/mbs1



loss1_test = np.zeros(max_itr)
loss2_test = np.zeros(max_itr)
loss1_test2 = np.zeros(max_itr)
loss2_test2 = np.zeros(max_itr)
loss1_test3 = np.zeros(max_itr)
loss2_test3 = np.zeros(max_itr)
loss1_vanilla = np.zeros(max_itr)
loss2_vanilla = np.zeros(max_itr)
loss1_LISTA = np.zeros(max_itr)
loss2_LISTA = np.zeros(max_itr)


for i in tqdm(range(mbs2)): #mbs2 = 100  #Aの数 ひとつのAに対して100個ｘを生成する
  x = gen_minibatch().detach().numpy()
  w = torch.normal(torch.zeros(mbs1, m), sigma).detach().numpy()
  A = torch.normal(torch.zeros(m, n), std = 1.0).detach().numpy() # 観測行列
  y = x@A.T + w
 
  alpha_ini = 1/np.max( np.abs( np.linalg.eigvals(A.T@A) ) )
  alpha2_test = np.loadtxt('alpha_DARTS_sup(try25.1).txt')
  
  #通常のISTA
  loss1_vanilla += vanilla_ISTA_MSE(y,A,x,alpha_ini, max_itr)
  loss2_vanilla += vanilla_ISTA_Fy(y,A,x,alpha_ini, max_itr)
  
  #alpha-LISTA
  loss1_LISTA += vanilla_LISTA_MSE(y,A,x,alpha2_test, max_itr)
  loss2_LISTA += vanilla_LISTA_Fy(y,A,x,alpha2_test, max_itr)
  
  #DARTS-ISTA
  alpha_test = np.loadtxt('alpha_DARTS_sup(try89.1).txt')
  #alpha2_test = np.loadtxt('alpha_DARTS_sup(try89.1).txt')
  
  beta1_test = np.loadtxt('beta1_DARTS_sup(try89.1).txt')
  beta2_test = np.loadtxt('beta2_DARTS_sup(try89.1).txt')
  gamma1_test = np.loadtxt('gamma1_DARTS_sup(try89.1).txt')
  gamma2_test = np.loadtxt('gamma2_DARTS_sup(try89.1).txt')
  loss1_test += ISTA_MSE(y,A,x,alpha_test, beta1_test,beta2_test,gamma1_test,gamma2_test,max_itr)
  loss2_test += ISTA_Fy(y,A,x,alpha_test, beta1_test,beta2_test,gamma1_test,gamma2_test,max_itr)
  
  #DARTS-ISTA
  alpha_test2 = np.loadtxt('alpha_DARTS_sup(try116.1).txt')
  #alpha2_test = np.loadtxt('alpha_DARTS_sup(try89.1).txt')
  
  beta1_test2 = np.loadtxt('beta1_DARTS_sup(try116.1).txt')
  beta2_test2 = np.loadtxt('beta2_DARTS_sup(try116.1).txt')
  gamma1_test2 = np.loadtxt('gamma1_DARTS_sup(try116.1).txt')
  gamma2_test2 = np.loadtxt('gamma2_DARTS_sup(try116.1).txt')
  loss1_test2 += ISTA_MSE2(y,A,x,alpha_test2, beta1_test2,beta2_test2,gamma1_test2,gamma2_test2,max_itr)
  loss2_test2 += ISTA_Fy2(y,A,x,alpha_test2, beta1_test2,beta2_test2,gamma1_test2,gamma2_test2,max_itr)
  
  #DARTS-ISTA
  alpha_test3 = np.loadtxt('alpha_DARTS_sup(try118.2).txt')
  #alpha2_test = np.loadtxt('alpha_DARTS_sup(try89.1).txt')
  
  beta1_test3 = np.loadtxt('beta1_DARTS_sup(try118.2).txt')
  beta2_test3 = np.loadtxt('beta2_DARTS_sup(try118.2).txt')
  gamma1_test3 = np.loadtxt('gamma1_DARTS_sup(try118.2).txt')
  gamma2_test3 = np.loadtxt('gamma2_DARTS_sup(try118.2).txt')
  loss1_test3 += ISTA_MSE3(y,A,x,alpha_test3, beta1_test3,beta2_test3,gamma1_test3,gamma2_test3,max_itr)
  loss2_test3 += ISTA_Fy3(y,A,x,alpha_test3, beta1_test3,beta2_test3,gamma1_test3,gamma2_test3,max_itr)

  
loss1_vanilla /= mbs2
loss1_test/= mbs2
loss1_test2/= mbs2
loss1_test3/= mbs2
loss1_LISTA/=mbs2

loss2_vanilla /= mbs2
loss2_test /= mbs2
loss2_test2 /= mbs2
loss2_test3 /= mbs2
loss2_LISTA /=mbs2

print(loss1_vanilla)
print(loss1_test)
print(loss1_test2)
print(loss1_test3)


#MSEのグラフ出力
fig, ax = plt.subplots(constrained_layout = True)
x_max = max_itr
plt.grid(which='major',color='black',linestyle='-')
plt.grid(which='minor',color='gray',linestyle='--')
plt.xlim(0,x_max)
# plt.ylim(1e-4,0.5)
plt.xticks([x_max*(0/5),x_max*(1/5),x_max*(2/5),x_max*(3/5),x_max*(4/5),x_max*(5/5)])
#ax.set_box_aspect(1)

ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('MSE',fontsize=20)
ax.tick_params(labelsize=20)

# ax.plot(loss1_vanilla,color=(76/255, 114/255, 176/255),label = 'ISTA',marker='s',markevery = 15,linestyle='--')
# # ax.plot(loss1_LISTA,color = (221/255, 132/255, 82/255),label = 'alpha-LISTA',marker='^',markevery = 15,linestyle='--')
# ax.plot(loss1_test,color=(85/255, 168/255, 104/255),label = 'AS-ISTA',marker='o',markevery = 15)
# ax.plot(loss1_test2,color=(196/255, 78/255, 82/255),label = 'BAS-ISTA(1)',marker='*',markevery = 15)
# ax.plot(loss1_test3,color=(129/255, 114/255, 179/255),label = 'BAS-ISTA(2)',marker='+',markevery = 15)
ax.plot(loss1_vanilla,color=(221/255, 132/255, 82/255),label = 'ISTA',marker='s',markevery = 15,linestyle='--')
# ax.plot(loss1_LISTA,color = (221/255, 132/255, 82/255),label = 'alpha-LISTA',marker='^',markevery = 15,linestyle='--')
ax.plot(loss1_test,color=(85/255, 168/255, 104/255),label = 'AS-ISTA',marker='o',markevery = 15)
ax.plot(loss1_test2,color=(196/255, 78/255, 82/255),label = 'BAS-ISTA(1)',marker='*',markevery = 15)
ax.plot(loss1_test3,color=(76/255, 114/255, 176/255),label = 'BAS-ISTA(2)',marker='+',markevery = 15)


plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.yscale('log')
plt.savefig("DARTS_MSE_L1_10xxx4_slide.pdf")

#目的関数の値のグラフ出力
fig, ax = plt.subplots(constrained_layout = True)

plt.grid(which='major',color='black',linestyle='-')
plt.grid(which='minor',color='gray',linestyle='--')
plt.xlim(0,max_itr)
plt.ylim(70,1e4)
plt.xticks([x_max*(0/5),x_max*(1/5),x_max*(2/5),x_max*(3/5),x_max*(4/5),x_max*(5/5)])
#ax.set_box_aspect(1)

ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('objective function',fontsize=20)
ax.tick_params(labelsize=20)

# ax.plot(loss2_vanilla,color = '#000000',label = 'ISTA',marker='s',markevery = 15)
# ax.plot(loss2_LISTA,color = 'blue',label = 'alpha-LISTA',marker='^',markevery = 15)
# ax.plot(loss2_test,color = 'red',label = 'AS-ISTA',marker='o',markevery = 15)
# ax.plot(loss2_test2,color = 'green',label = 'BAS-ISTA(1)',marker='*',markevery = 15)
# ax.plot(loss2_test3,color = 'purple',label = 'BAS-ISTA(2)',marker='+',markevery = 15)
ax.plot(loss2_vanilla,color=(76/255, 114/255, 176/255),label = 'ISTA',marker='s',markevery = 15,linestyle='--')
ax.plot(loss2_LISTA,color = (221/255, 132/255, 82/255),label = 'alpha-LISTA',marker='^',markevery = 15,linestyle='--')
ax.plot(loss2_test,color=(85/255, 168/255, 104/255),label = 'AS-ISTA',marker='o',markevery = 15)
ax.plot(loss2_test2,color=(196/255, 78/255, 82/255),label = 'BAS-ISTA(1)',marker='*',markevery = 15)
ax.plot(loss2_test3,color=(129/255, 114/255, 179/255),label = 'BAS-ISTA(2)',marker='+',markevery = 15)


plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.yscale('log')
plt.savefig("DARTS_Fy_L1_10xxx4.pdf")


#各反復でのステップサイズ
lip = 0
for gen in tqdm(range(max_itr)):
    A = torch.normal(torch.zeros(m, n), std = 1.0).to(device) # 観測行列
    D = A.detach().cpu().numpy()
    lip += 1/np.max( np.abs( np.linalg.eigvals(D.T@D) ) )
lip /= max_itr
lip = lip*np.ones(max_itr)

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
plt.xlim(0,max_itr)
plt.ylim(1e-7,0.5)
plt.xticks([0,20,40,60,80,100])
# plt.legend(bbox_to_anchor=(0, 0), loc='lower left', borderaxespad=0, fontsize=18)
# ax.set_box_aspect(1)
ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('step size',fontsize=20)
ax.tick_params(labelsize=20)
plt.yscale('log')

ax.plot(range(max_itr),lip,color=(76/255, 114/255, 176/255),label = 'ISTA',)
ax.plot(alpha2_test,color = (221/255, 132/255, 82/255),label = 'alpha-LISTA',)
ax.plot(alpha_test,color=(85/255, 168/255, 104/255),label = 'AS-ISTA',)
ax.plot(alpha_test2,color=(196/255, 78/255, 82/255),label = 'BAS-ISTA(1)')
ax.plot(alpha_test3,color=(129/255, 114/255, 179/255),label = 'BAS-ISTA(2)')

# plt.legend(bbox_to_anchor=(1, 1), loc='lower left', borderaxespad=0, fontsize=18)
plt.legend(loc='lower left', fontsize=16)


plt.savefig("alphaxxx4.pdf")

