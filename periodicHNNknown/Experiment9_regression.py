import os 
loc = os.getcwd()

import all_systems
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.prune as prune
import numpy as np
import os
import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks



error = lambda x,y: np.sum(np.sqrt(np.sum((x-y)**2,0))/np.sqrt(np.sum(x**2,0))) # where x is the true vector and y is the approximated vector

def sine_act(x, param1):
    return x + (param1)*(torch.sin(param1*x))*(torch.sin(param1*x))

# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)


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

class SnakeNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,periods):
        super(SnakeNet , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        self.periods= periods

    def forward(self, x):
        x = sine_act(self.hidden_layer_1(x), 2/self.periods)
        x = sine_act(self.hidden_layer_2(x), 2/self.periods) 
        x = self.output_layer(x)

        return x

class LearnNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LearnNet, self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size, bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

class SnakeParamNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,periods):
        super(SnakeParamNet, self).__init__()
        self.input_size = input_size 
        self.periods = periods
        self.hidden_layer_1 = nn.Linear( input_size*3, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        self.periods = periods
        
    def forward(self, x):
        outx = []
        for i in range(self.input_size):
          outx.append(x[:,i])
          outx.append(torch.cos(2/self.periods*x[:,i]))
          outx.append(torch.sin(2/self.periods*x[:,i]))
        x = torch.transpose(torch.vstack(outx),0,1)
        x = sine_act(self.hidden_layer_1(x), 2/self.periods)
        x = sine_act(self.hidden_layer_2(x), 2/self.periods)
        x = self.output_layer(x)
        return x

class ParamNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,periods):
        super(ParamNet , self).__init__()
        self.input_size = input_size
        
        self.hidden_layer_1 = nn.Linear( input_size*3, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        outx = []
        for i in range(self.input_size):
          outx.append(x[:,i])
          outx.append(torch.cos(2/periods*x[:,i]))
          outx.append(torch.sin(2/periods*x[:,i]))
        x = torch.transpose(torch.vstack(outx),0,1)
        x = softplus(self.hidden_layer_1(x))
        x = softplus(self.hidden_layer_2(x)) 
        x = self.output_layer(x)
        return x

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

df = pd.DataFrame(columns = ["seed","ini","model","loss","epoch","time"])

def get_grad(model, z,device):
  inputs=Variable(z.clone().detach()).requires_grad_(True).to(device)
  out=model(inputs.float())
  dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
  return np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2), int(len(z[0])))]), -np.asarray([dH.detach().cpu().numpy()[:,i] for i in range(int(len(z[0])/2))]) # negative dH/dq is dp/dt

n_sample = 10

totaldf = pd.DataFrame([])

for s in ["pendulum", "System3", "System5", "System11", "System12","doublepend"]: #["pendulum","trigo","arctan","System2","System3","System4","System5","System6","System7","System8","System9","System10","System11","System12","System13","System14"]: #["pendulum","trigo","arctan","System2","System3","System4","System5","System6","System7","System8","System9","System10","System11","System12","System13","System14"]:
    exec("sys = all_systems.%s" %s)
    print(s)
    dim = len(sys.spacedim)
    df = pd.DataFrame(columns = ["seed","ini","model","loss","epochs","time"])
    z = torch.tensor(np.array(np.meshgrid(*[np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) if sys.periods[i]>0 else np.linspace(sys.spacedim[i][0], sys.spacedim[i][1], 10) for i in range(dim) ],
                                      )))
    z = z.reshape(dim, int(torch.prod(torch.tensor(z.shape)).item()/dim)).transpose(1,0)

    for f in glob.glob(loc+"/Experiments/Baseline/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0] == "PINN"):
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "HNN"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})   
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)
              print(data)

    for f in glob.glob(loc+"/Experiments/LearningPeriodKnown/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if ((len(details) == 6) or (len(details) == 7)) & (details[0][:5] == "PINNL"):
            net = Net(sys.netshape1,sys.netshape2,1)
            net.load_state_dict(torch.load(f))
            model = "pHNN-L"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})   
            if (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    # for f in glob.glob(loc+"/Experiments/Observational/%s/*.pt" %s):
    for f in glob.glob(loc+"/Experiments/Observational/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "PINNO"):
            periods = np.asarray(sys.periods)
            periods = np.min(periods[np.nonzero(periods)])/(np.pi)
            name = details[0].split("-")
            multiplier = 3
            newwidth = int((-(sys.netshape1*multiplier+3)+np.sqrt((sys.netshape1*multiplier+3)**2 - 4*1*(sys.netshape1*2 - (sys.netshape1*sys.netshape2 + sys.netshape2*sys.netshape2 + sys.netshape2 + sys.netshape1 + sys.netshape2*2))))/(2))
            net = ParamNet(sys.netshape1,newwidth,1,sys.periods)
            net.load_state_dict(torch.load(f))
            model = "pHNN-O"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/InductiveSnakeLiu/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "sinsq"):
                periods = np.asarray(sys.periods)
                periods = np.min(periods[np.nonzero(periods)])/(np.pi)
                net = SnakeNet(sys.netshape1,sys.netshape2,1,periods)
                net.load_state_dict(torch.load(f))
                model = "pHNN-I"
                net = net.to(device)
                data = pd.Series({'seed':int(details[1]), 
                    'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                    'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                    'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                    "time":float(details[-1][:-3])})                   
                if  (int(details[3])>0):
                  df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/LO/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "PINNO"):
            periods = np.asarray(sys.periods)
            periods = np.min(periods[np.nonzero(periods)])/(np.pi)
            multiplier = 3
            newwidth = int((-(sys.netshape1*multiplier+3)+np.sqrt((sys.netshape1*multiplier+3)**2 - 4*1*(sys.netshape1*2 - (sys.netshape1*sys.netshape2 + sys.netshape2*sys.netshape2 + sys.netshape2 + sys.netshape1 + sys.netshape2*2))))/(2))
            net = ParamNet(sys.netshape1,newwidth,1,periods)
            net.load_state_dict(torch.load(f))
            model = "pHNN-OL"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/LI/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "sinsq"):
                net = SnakeNet(sys.netshape1,sys.netshape2,1,periods)
                periods = np.asarray(sys.periods)
                periods = np.min(periods[np.nonzero(periods)])/(np.pi)
                net.load_state_dict(torch.load(f))
                model = "pHNN-LI"
                net = net.to(device)
                data = pd.Series({'seed':int(details[1]), 
                    'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                    'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                    'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                    "time":float(details[-1][:-3])})       
            
                if  (int(details[3])>0):
                    df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/IO/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "PINNO"):
            name = details[0].split("-")
            multiplier = 3
            newwidth = int((-(sys.netshape1*multiplier+3)+np.sqrt((sys.netshape1*multiplier+3)**2 - 4*1*(sys.netshape1*2 - (sys.netshape1*sys.netshape2 + sys.netshape2*sys.netshape2 + sys.netshape2 + sys.netshape1 + sys.netshape2*2))))/(2))
            net = SnakeParamNet(sys.netshape1,newwidth,1,periods)
            net.load_state_dict(torch.load(f))
            model = "pHNN-IO"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)

    for f in glob.glob(loc+"/Experiments/ILO/%s/*.pt" %s):
        details = f.split("/")[-1].split("_")
        fvec = lambda z: np.concatenate([np.expand_dims(sys.f1(z),0), np.expand_dims(sys.f2(z),0)]) if len(sys.spacedim)==2 else np.concatenate([sys.f1(z), sys.f2(z)])
        if (len(details) == 5) & (details[0][:5] == "PINNO"):
            name = details[0].split("-")
            multiplier = 3
            newwidth = int((-(sys.netshape1*multiplier+3)+np.sqrt((sys.netshape1*multiplier+3)**2 - 4*1*(sys.netshape1*2 - (sys.netshape1*sys.netshape2 + sys.netshape2*sys.netshape2 + sys.netshape2 + sys.netshape1 + sys.netshape2*2))))/(2))
            net = SnakeParamNet(sys.netshape1,newwidth,1,periods)
            net.load_state_dict(torch.load(f))
            model = "pHNN-OLI"
            net = net.to(device)
            data = pd.Series({'seed':int(details[1]), 
                'ini':int(details[2]), 'model':model, 'epochs':int(details[3]),
                'loss':float(error(fvec(z), np.concatenate(get_grad(net, z, device)))/(n_sample**dim)), 
                'Hloss':torch.mean((sys.H(z.clone().detach()) - net(Variable(z.clone()).to(device).float()).detach().cpu()[:,0])**2),
                "time":float(details[-1][:-3])})         
            if  (int(details[3])>0):
              df = pd.concat([df,data.to_frame().T], ignore_index = True)





 



    df = df[df["ini"]==512]
    df["Experiment"] = s
    totaldf = pd.concat([totaldf, df]) 
totaldf = totaldf[totaldf['loss'].notna()]
plot_order = ["HNN", "pHNN-O", "pHNN-L", "pHNN-I", "pHNN-OL", "pHNN-LI", "pHNN-IO", "pHNN-OLI"] #["HNN", "HNN-O", "HNN-L", "HNN-I", "HNN-OL", "HNN-LI", "HNN-OI", "HNN-OLI"]

# view all
#totaldf = totaldf.replace({'Experiment' : { 'pendulum' : "Non-linear Pendulum", 'trigo' : 'Trigo', 'arctan' : 'Arctangent', 'System2' : 'System1', 'System3' : 'System2',
#			 'System4' : 'System3', 'System5' : 'System4', 'System6' : 'System5', 'System7' : 'System6', 
#			'System8' : 'System7', 'System9' : 'System8', 'System10' : 'System9', 'System11' : 'System10', 'System12' : 'System11', 'System13' : 'System12',
#			'System14' : 'System13',}})

totaldf = totaldf.replace({'Experiment' : { 'pendulum' : "Non-linear Pendulum", "System3": "System1", "System5": "System2",
                        "System11": "System3", "System12": "System4", 'doublepend' : "Double Pendulum",}})

totaldf["Vector Error"] = totaldf["loss"]*100
totaldf["Hamiltonian Error"] = totaldf["Hloss"].astype('float')
totaldf["Time"] = totaldf["time"]
totaldf["Epochs"] = totaldf["epochs"]

import matplotlib.ticker as mtick

sns.color_palette("tab10")

print(totaldf)
print(totaldf.groupby(["model","Experiment"]).mean())


import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

sns.set(font_scale=2.5)
fig, axes = plt.subplots(2,1, figsize = (24,13))
e = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"])], x="Experiment", y="Hamiltonian Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,
                order = ["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"], 
                ax=axes[0]); e.legend_.remove(); e.set_ylabel(r'$E_{H}$', rotation=90, labelpad=30); e.set_xlabel(None);
f = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"])], x="Experiment", y="Vector Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,
                order = ["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"],
                ax=axes[1]); f.set_ylabel(r'$E_{V}$', rotation=90, labelpad=30); f.set_xlabel(None); axes[1].yaxis.set_major_formatter(mtick.PercentFormatter());  f.legend_.remove();
"""
g = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"])], x="Experiment", y="Time", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,
                order = ["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"],
                ax=axes[2]); g.set_ylabel("Time", rotation=90, labelpad=30); g.legend_.remove(); g.set_xlabel(None);
h = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"])], x="Experiment", y="Epochs", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,
                order = ["Non-linear Pendulum","System1","System2","System3","System4","Double Pendulum"],
                ax=axes[3]); h.set_ylabel("Epochs", rotation=90, labelpad=30); h.legend_.remove(); h.set_xlabel(None);
"""
axes[0].set_yscale('log')
axes[1].set_yscale('log')
#axes[2].set_yscale('log')
#axes[3].set_yscale('log')
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.05), loc='center', ncol=4)
plt.savefig("Experiment9_5error_big.pdf", format="pdf", bbox_extra_artists=(lgd,), bbox_inches='tight') 
plt.clf()



"""
fig, axes = plt.subplots(3,1, figsize = (12,10.5))
e = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum",'Trigo','Arctangent',"System1","System2",])], x="Experiment", y="Vector Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order, ax=axes[0]); e.legend_.remove(); e.set(xlabel=None);
f = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System3","System4","System5","System6","System7","System8",])], x="Experiment", y="Vector Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[1]); f.legend_.remove(); f.set(xlabel=None);
g = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System9","System10","System11","System12","System13"])], x="Experiment", y="Vector Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[2]); g.legend_.remove(); 
handles, labels = axes[0].get_legend_handles_labels()
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')
lgd = fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.95), loc='upper right')
plt.savefig("periodicExperiment9_testvectorerror.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

fig, axes = plt.subplots(3,1, figsize = (12,10.5))
e = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum",'Trigo','Arctangent',"System1","System2",])], x="Experiment", y="Hamiltonian Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[0]); e.legend_.remove(); e.set(xlabel=None);
f = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System3","System4","System5","System6","System7","System8",])], x="Experiment", y="Hamiltonian Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[1]); f.legend_.remove(); f.set(xlabel=None);
g = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System9", "System10","System11","System12","System13"])], x="Experiment", y="Hamiltonian Error", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[2]); g.legend_.remove(); 
handles, labels = axes[0].get_legend_handles_labels()
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')
lgd = fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.95), loc='upper right')
plt.savefig("periodicExperiment9_testhamiltonianerror.png", bbox_extra_artists=(lgd,), bbox_inches='tight')


fig, axes = plt.subplots(3,1, figsize = (12,10.5))
e = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["Non-linear Pendulum",'Trigo','Arctangent',"System1","System2"])], x="Experiment", y="Time", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[0]); e.legend_.remove(); e.set(xlabel=None);
f = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System3","System4","System5","System6","System7","System8",])], x="Experiment", y="Time", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[1]); f.legend_.remove(); f.set(xlabel=None);
g = sns.barplot(data=totaldf[totaldf["Experiment"].isin(["System9", "System10","System11","System12","System13"])], x="Experiment", y="Time", hue="model", errorbar = "se", 
		palette = sns.color_palette("mako"), hue_order = plot_order,  ax=axes[2]); g.legend_.remove(); 
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.95), loc='upper right')
plt.savefig("periodicExperiment9_testtime.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
"""

"""

    df = df.groupby(by = ["ini","model"], as_index = False).agg(['mean','sem'])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df.reset_index()
    df = df[df["ini"]==512]
    mask = df.columns.str.contains('dim')
    df = df[["model","seed_mean","loss_mean","loss_sem","Hloss_mean","Hloss_sem","time_mean","epochs_mean"]+list(df.columns[mask])]
    print(s)
    print(df)

    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["model"]=="HNN"]["loss_mean"], df[df["model"]=="HNN"]["loss_sem"], 
								df[df["model"]=="sHNN-O"]["loss_mean"], df[df["model"]=="sHNN-O"]["loss_sem"],
								df[df["model"]=="sHNN-L"]["loss_mean"], df[df["model"]=="sHNN-L"]["loss_sem"],
								df[df["model"]=="sHNN-I"]["loss_mean"], df[df["model"]=="sHNN-I"]["loss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% " %(s, (1-df[df["model"]=="sHNN-O"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-L"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-I"]["loss_mean"].values/df[df["model"]=="HNN"]["loss_mean"].values)*100,))
    print("%s & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E) & %.2E (%.2E)" %(s, df[df["model"]=="HNN"]["Hloss_mean"], df[df["model"]=="HNN"]["Hloss_sem"], 
								df[df["model"]=="sHNN-O"]["Hloss_mean"], df[df["model"]=="sHNN-O"]["Hloss_sem"],
								df[df["model"]=="sHNN-L"]["Hloss_mean"], df[df["model"]=="sHNN-L"]["Hloss_sem"],
								df[df["model"]=="sHNN-I"]["Hloss_mean"], df[df["model"]=="sHNN-I"]["Hloss_sem"],))
    print("%s & 0.00 \\%% & %.2f \\%% & %.2f \\%% & %.2f \\%% " %(s, (1-df[df["model"]=="sHNN-O"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-L"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100, 
								(1-df[df["model"]=="sHNN-I"]["Hloss_mean"].values/df[df["model"]=="HNN"]["Hloss_mean"].values)*100,))
	
    print("%s & %.2f (%s) & %.2f (%s) & %.2f (%s) & %.2f (%s)" %(s, df[df["model"]=="HNN"]["time_mean"].values, df[df["model"]=="HNN"]["epochs_mean"].values[0], 
								df[df["model"]=="sHNN-O"]["time_mean"].values, df[df["model"]=="sHNN-O"]["epochs_mean"].values[0],
								df[df["model"]=="sHNN-L"]["time_mean"].values, df[df["model"]=="sHNN-L"]["epochs_mean"].values[0],
								df[df["model"]=="sHNN-I"]["time_mean"].values, df[df["model"]=="sHNN-I"]["epochs_mean"].values[0],))
"""



