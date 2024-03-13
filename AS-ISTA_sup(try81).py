#DARTSを実行

import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from matplotlib.animation import ArtistAnimation

m,n,k = 75,150,12  # 観測ベクトル次元 # 元信号ベクトル次元 #0成分の個数
snr = 10 #//雑音の分散関係
p = k/n  # 元信号の非ゼロ元の生起確率

sigma = np.sqrt(p/(10**(snr/10))) #雑音の分散 
lam = 10 #//ソフト閾値関数関係

max_itr = 100 #イテレーションの回数

device = torch.device('cuda') # 'cpu' or 'cuda'

#ハイパーパラメーター
mbs   = 30 # ミニバッチサイズ
adam_lr1 = 0.00007 # Adamの学習率
adam_lr2 = 0.007
para_num = 100  #パラメータの更新回数




A = torch.normal(torch.zeros(m, n), std = 1.0).detach().numpy() #m,nの行列をランダム生成numpy配列に格納
alpha_ini = 1/np.max( np.abs( np.linalg.eigvals(A.T@A) ) )*torch.ones(max_itr) #ISTAの式のα

#ミニバッチ生成関数
def gen_minibatch():
    seq = torch.normal(torch.zeros(mbs, n), 1.0) # ガウス乱数ベクトルの生成
    support = torch.bernoulli(p * torch.ones(mbs, n)) # 非ゼロサポートの生成
    return seq * support # 要素ごとの積(アダマール積)になることに注意


beta1_ini =10*torch.ones(max_itr) #すべて10の要素 構成要素はmax_itr個
beta2_ini =8.61*torch.ones(max_itr)
gamma1_ini =8.61*torch.ones(max_itr)
gamma2_ini =10*torch.ones(max_itr)

#学習型反復アルゴリズムのクラス定義
class ISTA(nn.Module):
    def __init__(self,max_itr):
        super(ISTA, self).__init__()
        self.alpha = nn.Parameter(alpha_ini) # 学習可能ステップサイズパラメータ
        self.beta1 = nn.Parameter(beta1_ini) #構造パラメータ
        self.beta2 = nn.Parameter(beta2_ini)
        self.gamma1 = nn.Parameter(gamma1_ini)
        self.gamma2 = nn.Parameter(gamma2_ini)
    
    #l1ノルムの近接写像
    def prox_L1(self, x, tau):
        return torch.sgn(x) * torch.maximum(torch.abs(x) - tau, torch.tensor((0)))

    def GD_step(self,x,tau): #勾配降下法
        return x + tau * (y - x @ A.t()) @ A
    
    def ST_step(self, x, tau): #ソフト閾値関数
        return self.prox_L1(x, lam * tau)

    #アルゴリズム本体
    def forward(self,num_itr):
        s = torch.zeros(mbs, n).to(device) # 初期探索点
        for i in range(num_itr):
            r = torch.exp(self.beta1[i])/(torch.exp(self.beta1[i])+torch.exp(self.beta2[i])) *self.GD_step(s, self.alpha[i]) + torch.exp(self.beta2[i])/(torch.exp(self.beta1[i])+torch.exp(self.beta2[i])) * self.ST_step(s, self.alpha[i])
            s = torch.exp(self.gamma1[i])/(torch.exp(self.gamma1[i])+torch.exp(self.gamma2[i])) * self.GD_step(r, self.alpha[i]) + torch.exp(self.gamma2[i])/(torch.exp(self.gamma1[i])+torch.exp(self.gamma2[i]))  * self.ST_step(r, self.alpha[i])
        return s


#オプティマイザ（Optimizer）は、機械学習およびディープラーニングにおいて、モデルのパラメータを調整し、訓練データに対する損失を最小化するためのアルゴリズムや方法です。モデルのパラメータを最適な値に調整するプロセスは、訓練または最適化として知られており、オプティマイザはこのプロセスを制御します。

model= ISTA(max_itr)
opt1 = optim.Adam(model.parameters(), lr=adam_lr1) #Adamオプティマイザの初期化:
opt2 = optim.Adam(model.parameters(), lr=adam_lr2)
loss_func = nn.MSELoss() #MSEの初期化
loss_MSE = np.zeros((max_itr*2)) #Numpy配列の初期化
loss_para = np.zeros((max_itr*para_num*2))
k,l = 0,0

for gen in tqdm(range(max_itr)):
    for j in range(2):
        if j == 0:
            model.alpha.requires_grad = True
            model.beta1.requires_grad = False
            model.beta2.requires_grad = False
            model.gamma1.requires_grad = False
            model.gamma2.requires_grad = False
        else:            
            model.alpha.requires_grad = False
            model.beta1.requires_grad = True
            model.beta2.requires_grad = True
            model.gamma1.requires_grad = True
            model.gamma2.requires_grad = True
        for i in range(para_num):
            x = gen_minibatch().to(device) # 元信号の生成
            w = torch.normal(torch.zeros(mbs, m), sigma).to(device)
            A = torch.normal(torch.zeros(m, n), std = 1.0).to(device) # 観測行列
            y = torch.mm(x, A.t()).to(device) + w # 観測信号の生成
            opt1.zero_grad() #zero
            opt2.zero_grad()
            x_hat = model(gen+1)
            loss = loss_func(x_hat, x)  #x_hatとxの誤差を計算     #教師あり学習
            loss.backward() #自動微分（Automatic Differentiation）による勾配計算を実行する重要なステップです。計算された損失に対して、PyTorchは誤差逆伝播（Backpropagation）を使用して、各パラメータに対する勾配を計算します。これにより、勾配情報がオプティマイザに渡され、モデルのパラメータが更新されます
            if j==0 : 
                opt1.step() #GDstep
                model.alpha.data.clamp_(min=0.0)
            else: 
                opt2.step()
            loss_para[k] = loss.item() #損失値を格納
            k = k + 1
        loss_MSE[l]=loss.item() #損失値を格納
        l=l+1
        if j == 0:  
            print("layer",gen+1," α learned", '{:.4e}'.format(loss.item()))
        else:  
            print("layer",gen+1,' β learned', '{:.4e}'.format(loss.item()))
        
# ステップサイズをテキストファイルで記録
np.savetxt('alpha_DARTS_sup(try90).txt',model.alpha.detach().numpy())
np.savetxt('beta1_DARTS_sup(try90).txt',model.beta1.detach().numpy())
np.savetxt('beta2_DARTS_sup(try90).txt',model.beta2.detach().numpy())
np.savetxt('gamma1_DARTS_sup(try90).txt',model.gamma1.detach().numpy())
np.savetxt('gamma2_DARTS_sup(try90).txt',model.gamma2.detach().numpy())

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
ax.set_box_aspect(1)
ax.set_xlabel('iteration',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(loss_para,color="red")
plt.yscale('log')
plt.savefig("loss_DARTS(try90).pdf")

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
ax.set_box_aspect(1)
ax.set_xlabel('iteration',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(loss_MSE,color="red")
plt.yscale('log')
plt.savefig("loss_MSE_DARTS(try90).pdf")