#goal is to finally get back in this and test if we can do all the biological plausibility enhancements necessary here without issue or not
#on this basis
#actually would be interesting to consider CIFAR but I don't tihnk that's likely
#first check is that backprop can do MNIST with just FC layers. This is pretty damn key overall, one must admit
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime

num_batches= 10
num_train_batches=20
batch_size = 64

def get_dataset(batch_size):
    #currently assuming just MNIST
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    trainset = list(iter(trainloader))
    testset = list(iter(testloader))
    for i,(img, label) in enumerate(trainset):
        trainset[i] = (img.reshape(len(img),784) /255 ,label)
    for i,(img, label) in enumerate(testset):
        testset[i] = (img.reshape(len(img),784) /255 ,label)
    return trainset, testset

#global DEVICE
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]


def onehot(x):
    z = torch.zeros([len(x),10])
    for i in range(len(x)):
      z[i,x[i]] = 1
    return z.float().to(DEVICE)


def set_tensor(xs):
  return xs.float().to(DEVICE)

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones((1,)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel 

def softmax(xs):
  return torch.nn.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))
   
def edge_zero_pad(img,d):
  N,C, h,w = img.shape 
  x = torch.zeros((N,C,h+(d*2),w+(d*2))).to(DEVICE)
  x[:,:,d:h+d,d:w+d] = img
  return x


def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B


class FCLayer(object):
  def __init__(self, input_size,output_size,batch_size, learning_rate,f,df,use_backwards_weights=True, use_backwards_nonlinearities=True,device="cpu"):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f 
    self.df = df
    self.device = device
    self.use_backwards_weights = use_backwards_weights
    self.use_backwards_nonlinearities = use_backwards_nonlinearities
    self.weights = torch.empty([self.input_size,self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
    if self.use_backwards_weights:
      self.backward_weights = torch.empty([self.output_size,self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self,x):
    self.inp = x.clone()
    self.activations = torch.matmul(self.inp, self.weights)
    return self.f(self.activations)

  def backward(self,e):
    self.fn_deriv = self.df(self.activations)
    if self.use_backwards_weights:
      if self.use_backwards_nonlinearities:
        out = torch.matmul(e * self.fn_deriv, self.backward_weights)
      else:
        out = torch.matmul(e, self.backward_weights)
    else:
      if self.use_backwards_nonlinearities:
        out = torch.matmul(e * self.fn_deriv, self.weights.T)
      else:
        out = torch.matmul(e, self.weights.T)
    return torch.clamp(out,-50,50)

  def update_weights(self,e,update_weights=False):
    self.fn_deriv = self.df(self.activations)
    if update_weights:
      if self.use_backwards_weights:
        if self.use_backwards_nonlinearities:
          delta = torch.matmul((e * self.fn_deriv).T,self.inp)
          dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        else:
          delta = torch.matmul(e.T, self.inp)
          dw = torch.matmul(self.inp.T, e)
        self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
        self.backward_weights += self.learning_rate * torch.clamp(delta*2,-50,50)
      else:
        if self.use_backwards_nonlinearities:
          dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        else:
          dw = torch.matmul(self.inp.T, e)
        self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))

class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate, weight_learning_rate,use_error_weights=False,update_error_connections=True,device='cpu'):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.weight_learning_rate = weight_learning_rate
    self.device = device
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.use_error_weights = use_error_weights
    self.update_error_connections = update_error_connections
    self.error_weights = []
    for i,l in enumerate(self.layers):
      if self.use_error_weights:
        #error_weight = set_tensor(torch.empty([l.input_size, l.input_size]).normal_(mean=0.0, std=0.05))
        #if i == 0:
        #  error_weight = set_tensor(torch.eye(l.input_size))
        #else:
          error_weight = (0.0 * set_tensor(torch.eye(l.input_size))) + set_tensor(torch.empty([l.input_size, l.input_size]).normal_(mean=0.0, std=0.05))
      else:
        error_weight = set_tensor(torch.eye(l.input_size))
      self.error_weights.append(error_weight)
    for l in self.layers:
      l.set_weight_parameters()

  def update_weights(self,print_weight_grads=True,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
      #true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
      #if print_weight_grads:
      #  diff = torch.sum((dW -true_dW)**2)
      #  weight_diffs.append(diff)
    return weight_diffs

  def update_error_weights(self):
    for (i,l) in enumerate(self.layers):
      if i != 0:
        #error_connection_delta = torch.matmul(self.outs[i].T,self.prediction_errors[i]) 
        error_connection_delta = torch.matmul(self.mus[i].T, self.prediction_errors[i]) #WORKING
        #print(error_connection_delta.shape)
        #error_connection_delta = torch.matmul(self.v_pred_errs[i+1],self.v_layers[i].mu.T)
        #self.error_weights[i] -=  self.weight_learning_rate * torch.clamp(error_connection_delta,-1,1) 
        self.error_weights[i] -=   self.weight_learning_rate * torch.clamp(error_connection_delta,-1,1)
        #print(self.error_weights[i])
        #print(self.error_weights[i])
        #pass

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def save_model(sef, savedir, logdir, losses,accs,test_accs):
    for i,l in enumerate(self.layers):
        l.save_layer(logdir,i)
    np.save(logdir +"/losses.npy",np.array(losses))
    np.save(logdir+"/accs.npy",np.array(accs))
    np.save(logdir+"/test_accs.npy",np.array(test_accs))
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])


  def infer(self, inp,label,n_inference_steps=None):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i,l in enumerate(self.layers):
        self.mus[i+1] = l.forward(self.mus[i])
        self.outs[i+1] = self.mus[i+1].clone()
      self.mus[-1] = label.clone()
      self.prediction_errors[-1] = self.mus[-1] - self.outs[-1] 
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
        for j in reversed(range(len(self.layers))):
          #if j != 0: 
          self.prediction_errors[j] = self.mus[j] - torch.matmul(self.outs[j],self.error_weights[j])
          self.prediction_errors[j] = torch.matmul(self.mus[j],self.error_weights[j]) -self.outs[j]
          self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
          dx_l = self.prediction_errors[j] - self.predictions[j]
          #print(dx_l.shape)
          self.mus[j] -= self.inference_learning_rate * (2*dx_l)
        #if self.use_error_weights: # so putting this here is still stable, which is odd. I'm not sure I feel like it should work with the capstone IF and only if it applies to the mus
        #but it's hard to figure out. But this will work even with the WRONG weight update
        #  self.update_error_weights()

      weight_diffs = self.update_weights()
      if self.use_error_weights and self.update_error_connections:
        self.update_error_weights()
      L = torch.sum(self.prediction_errors[-1]**2).item()
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def train(self,trainset, testset,logdir,savedir,n_epochs,n_inference_steps):
    with torch.no_grad():
        losses = []
        accs = []
        testaccs = []
        for epoch in range(n_epochs):
            losslist = []
            acclist = []
            print("Epoch: ", epoch)
            for i,(inp, label) in enumerate(trainset):
                L, acc,weight_diffs = self.infer(inp.to(DEVICE),onehot(label).to(DEVICE))
                print("Epoch: " + str(epoch) + " batch: " + str(i))
                print("Loss: ", L)
                print("Acc: ", acc)
                losslist.append(L)
                acclist.append(acc)
                #print("weight diffs: ", weight_diffs)
            losses = np.mean(np.array(losslist))
            accs = np.mean(np.array(acclist))
            mean_test_acc, _ = self.test_accuracy(testset)
            testaccs.append(mean_test_acc)
            self.save_model(logdir,savedir,losses,accs,testaccs)
            # so currently not even DOING test accuracy but I NEED to know this at any point.
            #to see if it's even feasible in the basic case. So let's do that.
            #Additionally, I need to start working with the qpreds libraries.


if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--learning_rate",type=float,default=0.0005)
    parser.add_argument("--N_epochs",type=int, default=1000)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--dataset",type=str,default="mnist")
    parser.add_argument("--use_backwards_weights",type=boolcheck, default=False)
    parser.add_argument("--use_backwards_nonlinearities",type=boolcheck, default=True)
    parser.add_argument("--use_error_connections",type=boolcheck,default=False)
    parser.add_argument("--update_error_connections",type=boolcheck,default=True)
    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    trainset,testset = get_dataset(args.batch_size)
    l1 = FCLayer(784,300,64,args.learning_rate,tanh,tanh_deriv,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,device=DEVICE)
    l2 = FCLayer(300,100,64,args.learning_rate,tanh,tanh_deriv,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,device=DEVICE)
    l3 = FCLayer(100,10,64,args.learning_rate,tanh,linear_deriv,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,device=DEVICE)
    layers =[l1,l2,l3]
    net = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,args.learning_rate,use_error_weights=args.use_error_connections,update_error_connections =args.update_error_connections, device=DEVICE)
    net.train(trainset[0:-2],testset[0:-2],args.logdir, args.savedir,args.N_epochs, args.n_inference_steps)