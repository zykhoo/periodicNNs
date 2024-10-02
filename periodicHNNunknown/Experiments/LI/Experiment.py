import os 
import sys
from pathlib import Path

sys.path.insert(0, '/home/ziyu/periodicHNNknown')
import all_systems

loc = os.getcwd()

import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from tqdm import tqdm
import math
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import random
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

deviceno = int(0)
if torch.cuda.is_available():
  device=torch.device('cuda:%s' %deviceno)
else:
  device=torch.device('cpu')

s = 'doublepend'
exec("sys = all_systems.%s" %s)

from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


c = '%sCheckpoint.pt' %s

initialcon = [512]


# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)


def sine_act(x, param1):
    return x + (param1)*(torch.sin(param1*x))*(torch.sin(param1*x))



# PINN
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

class LearnNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LearnNet, self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        x = sine_act(self.hidden_layer_1(x), self.alpha) # F.relu(self.hidden_layer_1(x)) # 
        x = sine_act(self.hidden_layer_2(x), self.alpha) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

class TrainNet(nn.Module):

    def __init__(self, input_size):
        super(TrainNet, self).__init__()
        self.trainable_params = nn.ParameterList([nn.Parameter(torch.ones(input_size)*2*math.pi)])
        
    def forward(self, x):
        outputs = list()
        for i in range(len(self.trainable_params[0])):
          outputs.append(torch.hstack((x[:,:i], x[:,i].reshape(-1,1) + torch.ones(x.shape[0],1).to(device)*self.trainable_params[0][i], x[:,(i+1):])))
          outputs.append(torch.hstack((x[:,:i], x[:,i].reshape(-1,1) - torch.ones(x.shape[0],1).to(device)*self.trainable_params[0][i], x[:,(i+1):])))
        return outputs 


class SnakeNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,param1, param2):
        super(SnakeNet , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = sine_act(self.hidden_layer_1(x),param1,param2)
        x = sine_act(self.hidden_layer_2(x),param1,param2) 
        x = self.output_layer(x)

        return x

class ParamNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ParamNet , self).__init__()
        self.input_size = input_size 
        self.periods = nn.Parameter(torch.ones(input_size))
        self.hidden_layer_1 = nn.Linear( input_size*3, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        outx = []
        for i in range(self.input_size):
          outx.append(x[:,i])
          outx.append(torch.cos(self.periods[i]*x[:,i]))
          outx.append(torch.sin(self.periods[i]*x[:,i]))
        x = torch.transpose(torch.vstack(outx),0,1)
        x = softplus(self.hidden_layer_1(x))
        x = softplus(self.hidden_layer_2(x)) 
        x = self.output_layer(x)
        return x

def lossfuc(model,mat,x,y,device,x0,H0,dim,c1=1,c2=1,c3=1,c4=1,epoch=0,periods=None,verbose=False):
    dim = int(wholemat.shape[1]/2)
    f3=(model(torch.tensor([[x0]*dim]).to(device))-torch.tensor([[H0]]).to(device))**2
    dH=torch.autograd.grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
    dHdq=dH[:,:int(dim/2)]
    dHdp=dH[:,int(dim/2):]
    qprime=(mat[:,dim:int(3*dim/2)])
    pprime=(mat[:,int(3*dim/2):])
    assert dHdq.shape[1] == int(dim/2)
    assert dHdp.shape[1] == int(dim/2)
    assert qprime.shape[1] == int(dim/2)
    assert pprime.shape[1] == int(dim/2)
    f1=torch.mean((dHdp-qprime)**2,dim=0)
    f2=torch.mean((dHdq+pprime)**2,dim=0)
    f4 = torch.zeros(1).to(device)

    x_period, y_period = periods


    for i in range(dim):
      if (i<int(dim/2)):
        #zeta_plus = Variable(torch.hstack((x.clone()[:,:i], x.clone()[:,i].reshape(-1,1) + torch.ones(x.shape[0],1).to(device)*model.trainable_params[0][i], x.clone()[:,(i+1):])).to(device).to(torch.float64),requires_grad=True).to(device)
        #zeta_minus = Variable(torch.hstack((x.clone()[:,:i], x.clone()[:,i].reshape(-1,1) - torch.ones(x.shape[0],1).to(device)*model.trainable_params[0][i], x.clone()[:,(i+1):])).to(device).to(torch.float64),requires_grad=True).to(device)

        #y_plus = model(zeta_plus)
        #y_minus = model(zeta_minus)
        dH_plus=torch.autograd.grad(y_period[i*2], x_period[i*2], grad_outputs=y_period[i*2].data.new(y_period[i*2].shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
        dH_minus=torch.autograd.grad(y_period[i*2+1], x_period[i*2+1], grad_outputs=y_period[i*2+1].data.new(y_period[i*2+1].shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
        dHdq_plus=dH_plus[:,i]
        dHdq_minus=dH_minus[:,i]
        f4+=torch.mean((dHdq_plus+mat[:,int(3*dim/2)+i])**2,dim=0) 
        f4+=torch.mean((dHdq_minus+mat[:,int(3*dim/2)+i])**2,dim=0) 
      elif (i>=int(dim/2)):
        #zeta_plus = Variable(torch.hstack((x.clone()[:,:i], x.clone()[:,i].reshape(-1,1) + torch.ones(x.shape[0],1).to(device)*model.trainable_params[0][i], x.clone()[:,(i+1):])).to(device).to(torch.float64),requires_grad=True).to(device)
        #zeta_minus = Variable(torch.hstack((x.clone()[:,:i], x.clone()[:,i].reshape(-1,1) - torch.ones(x.shape[0],1).to(device)*model.trainable_params[0][i], x.clone()[:,(i+1):])).to(device).to(torch.float64),requires_grad=True).to(device)
        #y_plus = model(zeta_plus)
        #y_minus = model(zeta_minus)
        dH_plus=torch.autograd.grad(y_period[i*2], x_period[i*2], grad_outputs=y_period[i*2].data.new(y_period[i*2].shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
        dH_minus=torch.autograd.grad(y_period[i*2+1], x_period[i*2+1], grad_outputs=y_period[i*2+1].data.new(y_period[i*2+1].shape).fill_(1),retain_graph=True,create_graph=True, allow_unused=True)[0]
        dHdp_plus=dH_plus[:,i]
        dHdp_minus=dH_minus[:,i]
        f4+=torch.mean((dHdp_plus-mat[:,int(dim/2)+i])**2,dim=0)
        f4+=torch.mean((dHdp_minus-mat[:,int(dim/2)+i])**2,dim=0)



    loss=torch.mean(c1*f1+c2*f2+c3*f3+c4*f4)
    # if loss > 1000: print("errors:", f1, f2, f3, f4)
    meanf1,meanf2,meanf3,meanf4=torch.mean(c1*f1),torch.mean(c2*f2),torch.mean(c3*f3),torch.mean(c4*f4)
    if verbose==True:
      print(x)
      print(meanf1,meanf2,meanf3,meanf4)
      print(loss,meanf1,meanf2,meanf3,meanf4)
    return loss,meanf1,meanf2,meanf3,meanf4


def data_preprocessing(start_train, final_train,device):       
    wholemat = np.hstack((start_train.transpose(), final_train.transpose()))
    wholemat = torch.tensor(wholemat)
    wholemat = wholemat.to(device)
    wholemat,evalmat=train_test_split(wholemat, train_size=0.8, random_state=1)
    return wholemat,evalmat


# evaluate loss of dataset 
def get_loss(model,device,initial_conditions,bs,x0,H0,dim,wholemat,evalmat,c1,c2,c3,c4,epoch,periods,trainset=False,verbose=False):
    # this function is used to calculate average loss of a whole dataset
    # rootpath: path of set to be calculated loss
    # model: model
    # trainset: is training set or not
    if trainset:
        mat=wholemat
    else:
        mat=evalmat
    avg_loss=0
    avg_f1=0
    avg_f2=0
    avg_f3=0
    avg_f4=0
    for count in range(0,len(mat),bs):
      curmat=mat[count:count+bs]
      x=Variable((curmat).to(torch.float64),requires_grad=True).to(device)[:,:dim]
      y=model(x)
      x_period = periods(x)
      y_period = [model(x_period[i]) for i in range(x.shape[1]*2)]


      loss,f1,f2,f3,f4=lossfuc(model,curmat,x,y,device,x0,H0,dim,c1,c2,c3,c4,epoch,[x_period, y_period],False)
      avg_loss+=loss.detach().cpu().item()
      avg_f1+=f1.detach().cpu().item()
      avg_f2+=f2.detach().cpu().item()
      avg_f3+=f3.detach().cpu().item()
      avg_f4+=f4.detach().cpu().item()
    num_batches=len(mat)//bs
    avg_loss/=num_batches
    avg_f1/=num_batches
    avg_f2/=num_batches
    avg_f3/=num_batches
    avg_f4/=num_batches
    if verbose == True:
        print(' loss=',avg_loss,' f1=',avg_f1,' f2=',avg_f2,' f3=',avg_f3,' f4=',avg_f4)
    return avg_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if abs(self.counter-self.patience)<5:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), c)
        self.val_loss_min = val_loss

def train(net,name,bs,num_epoch,initial_conditions,device,wholemat,evalmat,x0,H0,dim,LR,patience,c1,c2,c3,c4,periods):
    starttime = time.time() 
    # function of training process
    # net: the model
    # bs: batch size 
    # num_epoch: max of epoch to run
    # initial_conditions: number of trajectory in train set
    # patience: EarlyStopping parameter
    # c1~c4: hyperparameter for loss function

    smarker = 1
    avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li=[],[],[],[],[]
    avg_vallosses=[]
    
    start = time.time()
    lr = LR # initial learning rate
    net=net.to(device)
    periods=periods.to(device) 
    print(net.parameters(), periods.parameters())

    optimizer=torch.optim.Adam([*net.parameters(), *periods.parameters()], lr=LR )
    early_stopping = EarlyStopping(patience=patience, verbose=False,delta=0.00001) # delta


    for epoch in range(num_epoch):        

        running_loss=0

        running_f1=0
        running_f2=0
        running_f3=0
        running_f4=0
        num_batches=0
        
        # train
        shuffled_indices=torch.linspace(0,len(wholemat)-1,len(wholemat)).type(torch.long)
        net.train()
        periods.train()
        for count in range(0,len(wholemat),bs):
            optimizer.zero_grad()

            indices=shuffled_indices[count:count+bs]
            mat=wholemat[indices]

            x=Variable(torch.tensor(mat[:,:dim]).to(torch.float64),requires_grad=True)
            x_period = periods(x)
            y=net(x)
            y_period = [net(x_period[i]) for i in range(x.shape[1]*2)]

            loss,f1,f2,f3,f4=lossfuc(net,mat,x,y,device,x0,H0,dim,c1,c2,c3,c4,epoch,[x_period, y_period],False,) 
            loss.backward()
            if epoch < 4000: 
              torch.nn.utils.clip_grad_norm(net.parameters() , 1)
              torch.nn.utils.clip_grad_norm(periods.parameters() , 0.01)
            else:
              torch.nn.utils.clip_grad_norm(net.parameters(), 0.01)
              torch.nn.utils.clip_grad_norm(periods.parameters() , 0.01)

            optimizer.step()

            # compute some stats
            running_loss += loss.detach().item()
            running_f1 += f1.detach().item()
            running_f2 += f2.detach().item()
            running_f3 += f3.detach().item()
            running_f4 += f4.detach().item()

            num_batches+=1
            torch.cuda.empty_cache()



        avg_loss = running_loss/num_batches
        avg_f1 = running_f1/num_batches
        avg_f2 = running_f2/num_batches
        avg_f3 = running_f3/num_batches
        avg_f4 = running_f4/num_batches
        elapsed_time = time.time() - start
        
        avg_lossli.append(avg_loss)
        avg_f1li.append(avg_f1)
        avg_f2li.append(avg_f2)
        avg_f3li.append(avg_f3)
        avg_f4li.append(avg_f4)
        
        # evaluate
        net.eval()
        periods.eval()
        avg_val_loss=get_loss(net,device,len(evalmat),bs,x0,H0,dim,wholemat,evalmat,c1,c2,c3,c4,epoch,periods,False,)
        avg_vallosses.append(avg_val_loss)
        
  
        if epoch % 500 == 0 : 
            # print(' ') 
                       
          print('epoch=',epoch, ' time=', elapsed_time, 
                  ' loss=', avg_loss ,' val_loss=',avg_val_loss,' f1=', avg_f1 ,' f2=', avg_f2 ,
                  ' f3=', avg_f3 ,' f4=', avg_f4 , 'num_batches=', num_batches, "alpha=", net.alpha, "periods=", periods.trainable_params[0], 'percent lr=', optimizer.param_groups[0]["lr"])

        if time.time() - starttime > smarker:
            torch.save(net.state_dict(), "%s_%s_%s.pt" %(name,epoch,time.time()-starttime))
            smarker += 20
        
        if epoch%100 == 0:
            torch.save(net.state_dict(), c)
        
        if math.isnan(running_loss):
            text_file = open("nan_report.txt", "w")
            text_file.write('name=%s at epoch %s' %(name, epoch))
            text_file.close()
            print("saving this file and ending the training")
            net.load_state_dict(torch.load(c)) 
            return net,epoch,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li

        
        

        early_stopping(avg_val_loss,net)
        if early_stopping.early_stop:
              print('Early Stopping')
              break
            
    net.load_state_dict(torch.load(c)) #net=torch.load(c)
    return net,epoch,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li,periods.trainable_params[0]


def CreateTrainingDataExact(traj_len,ini_con,spacedim,h,f1,f2,seed,n_h = 800,t=None):
  np.random.seed(seed = seed)
  start = np.vstack([np.random.uniform(low = spacedim[i][0], high = spacedim[i][1], size = ini_con) for i in range(len(spacedim))])
  f = lambda x: np.expand_dims(np.hstack([f1(x), f2(x)]),1) 
  delta = f(start[:,0])
  for k in range(ini_con-1):
    new_delta = f(np.squeeze(start[:,k+1]))
    delta = np.hstack((delta, new_delta))
  return start, delta

def weights_copy(copyto, copyfrom):
  with torch.no_grad():	
    copyto.hidden_layer_1.weight.copy_(copyfrom.hidden_layer_1.weight)
    copyto.hidden_layer_1.bias.copy_(copyfrom.hidden_layer_1.bias)
    copyto.hidden_layer_2.weight.copy_(copyfrom.hidden_layer_2.weight)
    copyto.hidden_layer_2.bias.copy_(copyfrom.hidden_layer_2.bias)
    copyto.output_layer.weight.copy_(copyfrom.output_layer.weight)
    copyto.output_layer.bias.copy_(copyfrom.output_layer.bias)
  return copyto


for i in range(5):
  seed = i
  np.random.seed(seed=seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  for ini in initialcon: 

    x0, H0, LR, h, f1, f2, dim, periods = sys.x0, sys.H0, sys.LR, sys.h, sys.f1gen, sys.f2gen, len(sys.spacedim), sys.periods


    #start, delta = CreateTrainingDataExact(1,ini,sys.spacedim,h,f1,f2,seed = seed,n_h = 1,t=None)
    

    #wholemat, evalmat = data_preprocessing(start, delta, device) 
    wholemat = torch.tensor(np.loadtxt(str(Path(loc).parents[0]) + '/Baseline/data/%s_%s_wholemat.txt' %(s,seed))).to(device)
    evalmat = torch.tensor(np.loadtxt(str(Path(loc).parents[0]) + '/Baseline/data/%s_%s_evalmat.txt' %(s,seed))).to(device)

    periods = TrainNet(sys.netshape1)
    orinet = Net(sys.netshape1, sys.netshape2, 1)
    net = LearnNet(sys.netshape1,sys.netshape2,1)
    starttime = time.time() 
    print("training PINN Net")
    orinet.load_state_dict(torch.load('%s/%s/%s_%s_%s_%s_%s.pt' %(str(Path(loc).parents[0])+ '/Baseline',s,"PINN",seed,ini,0,0)))
    weights_copy(net, orinet)
    torch.save(net.state_dict(), '%s/%s/%s_%s_%s_%s_%s_%s.pt' %(loc,s,"PINNL",seed,ini,0,0,4000))
    results = train(net,name="%s/%s/%s_%s_PINNL_%s" %(loc,s,seed,ini,4000),bs=int(len(wholemat)/5),num_epoch=150001,initial_conditions=initialcon,device=device, wholemat=wholemat,evalmat=evalmat,x0=x0,H0=H0,dim=dim,LR=LR,patience=4000,c1=1,c2=1,c3=1,c4=1,periods=periods)
    net, epochs, periods = results[0], results[1], results[-1]
    periods_string = "-".join(format(x, "10.5f") for x in periods)
    PINNtraintime = time.time()-starttime
    torch.save(net.state_dict(), '%s/%s/%s_%s_%s_%s_%s_%s_%s.pt' %(loc,s,"PINNL",seed,ini,epochs,4000,periods_string,PINNtraintime))

