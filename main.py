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

def get_dataset(batch_size,norm_factor):
    #currently assuming just MNIST
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                            download=False, transform=transform)
    print("trainset: ", trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False)
    print("trainloader: ", trainloader)
    trainset = list(iter(trainloader))

    testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    for i,(img, label) in enumerate(trainset):
        trainset[i] = (img.reshape(len(img),784) /norm_factor ,label)
    for i,(img, label) in enumerate(testset):
        testset[i] = (img.reshape(len(img),784) /norm_factor ,label)
    return trainset, testset


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

def parse_activation_functions(act_fn):
  if act_fn == "relu":
    f = relu
    df = relu_deriv
  elif act_fn == "tanh":
    f = tanh
    df = tanh_deriv
  elif act_fn == "sigmoid":
    f = sigmoid
    df = sigmoid_deriv
  else:
    raise ValueError("Activation function not recognised. Available activation functions are: 'relu', 'tanh','sigmoid'.")
  return f, df

def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B


class FCLayer(object):
  def __init__(self, input_size,output_size,batch_size, learning_rate,f,df,use_backwards_weights=True, use_backwards_nonlinearities=True,use_bias=False,weight_decay_coeff=0,weight_normalization=False,device="cpu"):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f 
    self.df = df
    self.device = device
    self.use_backwards_weights = use_backwards_weights
    self.use_backwards_nonlinearities = use_backwards_nonlinearities
    self.weight_decay_coeff = weight_decay_coeff
    self.weight_normalization = weight_normalization
    self.weights = torch.empty([self.input_size,self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
    self.use_bias = use_bias
    self.bias = torch.zeros([self.batch_size, self.output_size]).to(self.device)
    if self.use_backwards_weights:
      self.backward_weights = torch.empty([self.output_size,self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)

  def forward(self,x):
    self.inp = x.clone()
    self.activations = torch.matmul(self.inp, self.weights) 
    return self.f(self.activations) + self.bias

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
    if self.use_backwards_weights:
      delta = torch.matmul((e * self.fn_deriv).T,self.inp)
      dw = torch.matmul(self.inp.T, e * self.fn_deriv)
      if update_weights:
        self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
        self.backward_weights += self.learning_rate * torch.clamp(delta*2,-50,50)
    else:
      dw = torch.matmul(self.inp.T, e * self.fn_deriv)
      if update_weights:
        dW  = torch.clamp(dw*2,-50,50)
        self.weights += (self.learning_rate * (dW ))#- torch.mean(dW))) #- (torch.mean(torch.abs(dw)) * self.weights)#(self.weight_decay_coeff * self.weights)
        #if self.weight_normalization:
        #  self.weights = set_tensor(torch.ones_like(self.weights)) + (self.weights - torch.mean(self.weights)) / torch.var(self.weights)
    #print("TOTAL DW: ", torch.sum(dw))
    #update bias
    if self.use_bias:
      self.bias += self.learning_rate * torch.clamp(e,-50,50)
    return dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate, weight_learning_rate,use_error_weights=False,with_amortisation=True, fixed_predictions=True, dynamical_weight_update=False, dilation_factor=20,enforce_negative_errors=False,error_weight_std=0.05,update_error_weights=True,device='cpu'):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.weight_learning_rate = weight_learning_rate
    self.device = device
    self.with_amortisation = with_amortisation
    self.fixed_predictions = fixed_predictions
    self.dynamical_weight_update = dynamical_weight_update
    self.dilation_factor = dilation_factor
    self.enforce_negative_errors = enforce_negative_errors
    self.error_weight_std = error_weight_std
    self.update_error_weights_flag = update_error_weights
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.use_error_weights = use_error_weights
    self.e_ys  = [[] for i in range(len(self.layers)+1)]
    self.xs  = [[] for i in range(len(self.layers)+1)]
    self.error_weights = []
    for i,l in enumerate(self.layers):
      if self.use_error_weights:
          error_weight = (0.0 * set_tensor(torch.eye(l.input_size))) + set_tensor(torch.empty([l.input_size, l.input_size]).normal_(mean=0.0, std=self.error_weight_std))
          if self.enforce_negative_errors is True:
            error_weight = torch.abs(error_weight)
      else:
        error_weight = set_tensor(torch.eye(l.input_size))
      self.error_weights.append(error_weight)
    #final output error weight which is always identity
    self.error_weights.append(set_tensor(torch.eye(self.layers[-1].output_size)))
    for l in self.layers:
      l.set_weight_parameters()
    if self.dynamical_weight_update:
      for l in self.layers:
        if hasattr(l, "learning_rate"):
          l.learning_rate = l.learning_rate / self.dilation_factor

  def BP_forward(self, inp):
    self.xs[0] = inp
    for i,l in enumerate(self.layers):
      self.xs[i+1] = l.forward(self.xs[i])
    return self.xs[-1]

  def BP_backward(self, e_y):
    self.e_ys[-1] = e_y
    for (i,l) in reversed(list(enumerate(self.layers))):
      self.e_ys[i] = l.backward(self.e_ys[i+1])
    return self.e_ys[0]

  def update_weights(self,print_weight_grads=True,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
      #true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
      #print(dW.shape)
      #print("dW: ", dW*2)
      #print("true grad: ", l.get_true_weight_grad())
      #if print_weight_grads:
      #  diff = torch.sum((dW -true_dW)**2)
      #  weight_diffs.append(diff)
    return weight_diffs

  def update_error_weights(self):
    #print("updating error weights")
    for (i,l) in enumerate(self.layers):
      if i != 0:
        error_connection_delta = torch.matmul(self.outs[i].T,self.prediction_errors[i]) 
        #error_connection_delta = torch.matmul(self.mus[i].T, self.prediction_errors[i]) #WORKING
        #print(error_connection_delta.shape)
        #error_connection_delta = torch.matmul(self.v_pred_errs[i+1],self.v_layers[i].mu.T)
        #self.error_weights[i] -=  self.weight_learning_rate * torch.clamp(error_connection_delta,-1,1) 
        #old_error_weights = self.error_weights[i].clone()
        #print("weight learning rate : ", self.weight_learning_rate)
        self.error_weights[i] = self.error_weights[i].clone() - (self.weight_learning_rate * torch.clamp(error_connection_delta,-50,50))
        #print(self.error_weights[i])
        #print("error connection delta: ", error_connection_delta)
        #print(self.error_weights[i])
        #print("Update diff: ", torch.sum(torch.abs(old_error_weights - self.error_weights[i])).item())
        #pass

  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def BP_pass(self, inp, label):
    out = self.BP_forward(inp)
    e_y = out - label
    self.BP_backward(e_y)

  def cosine_similarity(self,pc_e, bp_e):
    #pc_e = torch.randn_like(pc_e)
    #bp_e = torch.randn_like(bp_e)
    cos = torch.bmm(-bp_e.unsqueeze(2).permute(1,2,0),pc_e.permute(1,0).unsqueeze(2)).reshape([len(bp_e.T)])
    pc_norm = torch.norm(pc_e, p=2,dim=0)
    bp_norm = torch.norm(bp_e,p=2,dim=0)
    norm = pc_norm * bp_norm
    similarity =cos / norm
    return torch.mean(rad_to_degrees(torch.acos(similarity))).item()

  def cosine_similarities(self):
    similarities = []
    for (i,l) in enumerate(self.layers):
      if i != 0:
        #print("diff: ",torch.sum(torch.abs(self.prediction_errors[i] + self.e_ys[i])))
        similarities.append(self.cosine_similarity(self.prediction_errors[i],self.e_ys[i]))
    return similarities

  def weight_norm(self):
    norms = []
    for l in self.layers:
      norms.append(torch.sum(torch.square(l.weights)).item())
    return norms

  def save_model(self, savedir, logdir, losses,accs,test_accs):
    #for i,l in enumerate(self.layers):
    #    l.save_layer(logdir,i)
    np.save(logdir +"/losses.npy",np.array(losses))
    np.save(logdir+"/accs.npy",np.array(accs))
    np.save(logdir+"/test_accs.npy",np.array(test_accs))
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
    print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))
    subprocess.call(['echo','saved at time: ' + str(current_time)])

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def infer_pc(self):
    #calculate initial errors
    for i in range(1,len(self.layers)+1):
      self.outs[i] = self.layers[i-1].forward(self.mus[i-1])
      #self.prediction_errors[i] = (self.mus[i] @ self.error_weights[i]) - self.outs[i]
      self.prediction_errors[i] = self.mus[i] - (self.outs[i] @ self.error_weights[i])
      #print(torch.mean(torch.abs(self.prediction_errors[i])))
    for n in range(self.n_inference_steps_train):
      #update variable nodes
      for j in range(1,len(self.layers)):
        self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
        dx_l = self.prediction_errors[j] - self.predictions[j]
        self.mus[j] -= self.inference_learning_rate * (2*dx_l)
        #self.prediction_errors[j] = self.mus[j] - self.outs[j]
        #print("pe: ", self.prediction_errors[j])
      for i in range(1,len(self.layers)+1):
        if not self.fixed_predictions:
          self.outs[i] = self.layers[i-1].forward(self.mus[i-1])
        #self.prediction_errors[i] = (self.mus[i] @ self.error_weights[i]) - self.outs[i]
        self.prediction_errors[i] = self.mus[i] - (self.outs[i] @ self.error_weights[i])

      if self.dynamical_weight_update:
        weight_diffs = self.update_weights()
    #print("pe norm: ", [torch.sum(torch.square(pe)).item() for pe in self.prediction_errors[1:]])
    print([torch.mean(torch.abs(pe)) for pe in self.prediction_errors[1:]])

  def learn_pc(self, inp, label,backprop_pass_comparison=True):
    with torch.no_grad():
      if backprop_pass_comparison:
        self.BP_pass(inp, label)
      self.mus[0] = inp.clone()
      for i in range(1,len(self.layers)):
        self.mus[i] = self.layers[i-1].forward(self.mus[i-1])
        if not self.with_amortisation:
          print("replacing mu")
          self.mus[i] = set_tensor(torch.empty_like(self.mus[i]).normal_(mean=0,std=0.05)) 

      self.mus[-1] = label.clone()
      self.infer_pc()
      weight_diffs = self.update_weights()
      #print("IN LEARN PC UPDATE ERROR WEIGHTS" + str(self.use_error_weights) + " " + str(self.update_error_weights_flag))
      if self.use_error_weights and self.update_error_weights_flag:
        #print("UPDATING ERROR WEIGHTS!!!!!")
        self.update_error_weights()
      if backprop_pass_comparison:
        #print("cosine similarities: ", self.cosine_similarities())
        #print("DIFFS: ", [torch.mean(torch.abs(self.predictions[i] + self.e_ys[i])).item() for (i,l) in enumerate(self.layers) if i != 0])
        #print("Weight norm: ", self.weight_norm())
        pass
      #print("outs: ", self.outs[-1][0,:])
      #print("label: ", label[0,:])
      L = torch.sum(self.prediction_errors[-1]**2).item()
      #print("Prediction errors: ", self.prediction_errors[-1][0,:])
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def infer(self, inp,label,n_inference_steps=None,backprop_pass_comparison=True):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    if backprop_pass_comparison:
      self.BP_pass(inp, label)
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i,l in enumerate(self.layers):
        if self.with_amortisation:
          self.outs[i+1] = l.forward(self.outs[i])
          self.mus[i+1] = self.outs[i+1].clone()
        else:
          self.outs[i+1] = l.forward(self.outs[i].clone())
          self.mus[i+1] = set_tensor(torch.empty_like(self.outs[i+1]).normal_(mean=0,std=0.05)) 
      self.mus[-1] = label.clone()
      self.prediction_errors[-1] = label - self.outs[-1] 
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
        self.mus[0] = inp.clone()
        self.outs[0] = inp.clone()
        if not self.fixed_predictions:
          for i,l in enumerate(self.layers):
            self.outs[i+1] = l.forward(self.mus[i].clone())
        self.prediction_errors[-1] = label - self.outs[-1] #setup final prediction errors
        for j in reversed(range(len(self.layers))):
          self.prediction_errors[j] = torch.matmul(self.mus[j],self.error_weights[j]) -self.outs[j]
          print(self.prediction_errors[j].shape)
          self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
          dx_l = self.prediction_errors[j] - self.predictions[j]
          self.mus[j] -= self.inference_learning_rate * (2*dx_l)

        if self.dynamical_weight_update:
          weight_diffs = self.update_weights()
          if self.use_error_weights:
            self.update_error_weights()

      weight_diffs = self.update_weights()
      if self.use_error_weights and self.update_error_weights:
        self.update_error_weights()
      if backprop_pass_comparison:
        #print("cosine similarities: ", self.cosine_similarities())
        #print("DIFFS: ", [torch.mean(torch.abs(self.predictions[i] + self.e_ys[i])).item() for (i,l) in enumerate(self.layers)])
        pass
      #print("outs: ", self.outs[-1][0,:])
      #print("label: ", label[0,:])
      L = torch.sum(self.prediction_errors[-1]**2).item()
      #print("Prediction errors: ", self.prediction_errors[-1][0,:])
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def train(self,trainset,testset, logdir, savedir,n_epochs,n_inference_steps):
    with torch.no_grad():
      losses = []
      accs = []
      test_accs = []
      for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        for i,(inp, label) in enumerate(trainset):
          L, acc,weight_diffs = self.learn_pc(inp.to(DEVICE),onehot(label).to(DEVICE))
          print("Epoch: " + str(epoch) + " batch: " + str(i))
          print("Loss: ", L)
          print("Acc: ", acc)
          losses.append(L)
          accs.append(acc)
        
        mean_test_acc, _ = self.test_accuracy(testset)
        test_accs.append(mean_test_acc)
        self.save_model(logdir,savedir,losses,accs,test_accs)




if __name__ == '__main__':
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--norm_factor",type=int,default=1)
    parser.add_argument("--learning_rate",type=float,default=0.001)
    parser.add_argument("--N_epochs",type=int, default=10)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=500)
    parser.add_argument("--inference_learning_rate",type=float,default=0.01)
    parser.add_argument("--dataset",type=str,default="mnist")
    parser.add_argument("--use_backwards_weights",type=boolcheck, default=False)
    parser.add_argument("--use_backwards_nonlinearities",type=boolcheck, default=True)
    parser.add_argument("--use_error_weights",type=boolcheck,default=False)
    parser.add_argument("--update_error_weights",type=boolcheck,default=True)
    parser.add_argument("--enforce_negative_errors",type=boolcheck,default=True)
    parser.add_argument("--with_amortisation",type=boolcheck,default=True)
    parser.add_argument("--fixed_predictions",type=boolcheck,default=True)
    parser.add_argument("--dynamical_weight_update",type=boolcheck, default=False)
    parser.add_argument("--weight_decay_coeff",type=float,default=0.01)
    parser.add_argument("--weight_normalization",type=boolcheck,default=False)
    parser.add_argument("--use_bias",type=boolcheck,default=True)
    parser.add_argument("--activation_function",type=str,default="relu")
    args = parser.parse_args()
    print("Args parsed")
    #create folders
    if args.savedir != "":
        subprocess.call(["mkdir","-p",str(args.savedir)])
    if args.logdir != "":
        subprocess.call(["mkdir","-p",str(args.logdir)])
    print("folders created")
    trainset,testset = get_dataset(args.batch_size,args.norm_factor)
    # parse activation functions
    f,df = parse_activation_functions(args.activation_function)
    

    l1 = FCLayer(784,300,args.batch_size,args.learning_rate,f,df,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,weight_decay_coeff=args.weight_decay_coeff,weight_normalization = args.weight_normalization,use_bias = args.use_bias,device=DEVICE)
    l2 = FCLayer(300,100,args.batch_size,args.learning_rate,f,df,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,weight_decay_coeff=args.weight_decay_coeff,weight_normalization = args.weight_normalization,use_bias = args.use_bias,device=DEVICE)
    l3 = FCLayer(100,100,args.batch_size,args.learning_rate,f,df,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,weight_decay_coeff=args.weight_decay_coeff,weight_normalization = args.weight_normalization,use_bias = args.use_bias,device=DEVICE)
    l4 = FCLayer(100,10,args.batch_size,args.learning_rate,linear,linear_deriv,use_backwards_weights= args.use_backwards_weights, use_backwards_nonlinearities=args.use_backwards_nonlinearities,weight_decay_coeff=args.weight_decay_coeff,weight_normalization = args.weight_normalization,use_bias = args.use_bias,device=DEVICE)
    layers =[l1,l2,l3,l4]
    net = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,args.learning_rate,use_error_weights=args.use_error_weights,with_amortisation=args.with_amortisation, fixed_predictions=args.fixed_predictions,dynamical_weight_update=args.dynamical_weight_update,update_error_weights=args.update_error_weights,enforce_negative_errors=args.enforce_negative_errors,device=DEVICE)
    net.train(trainset[0:-2],testset[0:-2],args.logdir, args.savedir, args.N_epochs, args.n_inference_steps)
