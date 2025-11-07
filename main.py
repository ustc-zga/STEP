import os
from os.path import join as opj
from os.path import dirname as opd


import random
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn
from model import Model
import torch
import numpy as np
import argparse
import time
import shutil
from opt_type import SSHopt

from utils import gumbel_softmax
from metric import *
from dataloader import *


class Engine(object):
    def __init__(self, args: SSHopt.SSHargs, device="cuda"):
       
        self.args = args
        self.device = device
       

        
        self.model = Model(self.device, self.args.num_nodes, self.args.dropout, 
                        in_dim=self.args.in_dim, out_dim=self.args.seq_length, 
                        residual_channels=self.args.nhid, dilation_channels=self.args.nhid, 
                        skip_channels=self.args.nhid *4, end_channels=self.args.nhid *8,num_for_predict=self.args.seq_length,latend_num=self.args.group_nodes,num_layer=self.args.num_layer).to(self.device)
        

        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(self.model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        lr_schedule_length = self.args.total_epoch
        # gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        # self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(self.data_pred_optimizer, step_size=1, gamma=gamma)
        
        self.clip = 5
       
      
        self.graph = nn.Parameter(torch.ones([self.args.group_nodes, self.args.num_nodes,self.args.input_step]).to(self.device)*0)
        self.graph_optimizer = torch.optim.Adam([self.graph], lr=self.args.graph_discov.lr_graph_start)
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)

        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        
        end_lmd, start_lmd = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd
      
        self.loss = masked_mae
        
        
    
    def sample_bernoulli_time(self,sample_matrix):
        sample_matrix = torch.sigmoid(sample_matrix[None].expand(self.args.batch_size, -1, -1,-1))
        return torch.bernoulli(sample_matrix)
    
    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self.device) for tensor in tensors]
        else:
            return tensors.to(self.device)
        
    def train_pred(self,input,real_val,loss_weight):
        
       
        self.model.train()
        self.data_pred_optimizer.zero_grad()
        
     
        graph_sampled = self.sample_bernoulli_time(self.graph)[0]
         
        output,s_loss,A= self.model(input,graph_sampled)
        output = output.transpose(1,3)
        
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        pred_loss = self.loss(predict, real,0.0)#+energy
        
    
        loss=loss_weight[0]*pred_loss+loss_weight[1]*s_loss
        loss.backward()
        
      
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
        self.data_pred_optimizer.step()
        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        
        return loss.item(),mae,mape,rmse
    
    
    def graph_discov(self, input,real_val,loss_weight): 

      
        
        def gumbel_sigmoid_sample_time(graph, batch_size, tau=1):
            prob = torch.sigmoid(graph[None, :, :,:, None].expand(batch_size, -1, -1, -1,-1))
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :,:, 0]
            return samples
        
     
        # self.fitting_model.eval()
        self.graph_optimizer.zero_grad()
        
        prob_graph = torch.sigmoid(self.graph[None, :, :])
        graph_sampled= gumbel_sigmoid_sample_time(self.graph, self.args.batch_size, tau=self.gumbel_tau)[0]
        gs = prob_graph.shape
        
        output,s_loss,A= self.model(input, graph_sampled)
        
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        pred_loss = self.loss(predict, real,0.0)#+energy
        loss_sparsity = torch.norm(prob_graph, p=1) / (gs[0] * gs[1]*gs[2])
    
        
        loss = loss_weight[0]*pred_loss+self.lambda_s*loss_sparsity 
        
        loss.backward()
        self.graph_optimizer.step()
       
        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        
        return loss.item(),mae,mape,rmse
    
    def eval(self, input, real_val):
        self.model.eval()
     
    
        graph_sampled = self.sample_bernoulli_time(self.graph)[0]
           
        output,s_loss,A= self.model(input, graph_sampled)
        
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        pred_loss = self.loss(predict, real,0.0)
        
        
        loss=pred_loss
        
        
        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        
        return loss.item(),mae,mape,rmse     
    
    
    def train(self,loss_weights):
        
        device = torch.device(self.device)
  
        min_loss=np.inf
        dataloader = load_dataset(self.args.data, self.args.batch_size, self.args.batch_size, self.args.batch_size)
        params_path=self.args.check_save

        
        if os.path.exists(params_path) and not self.args.force:
           raise SystemExit("Params folder exists! Select a new params path please!")
        else:
            if os.path.exists(params_path):
                shutil.rmtree(params_path)
            os.makedirs(params_path)
         
        print('Create params directory %s' % (params_path))

        print("start training...",flush=True)
       
        val_loss =[]
        val_time = []
        val_mae=[]
        train_time = []   
        count=0      
      
        for i in range(1,self.args.total_epoch+1):
            train_loss = []
            train_mae = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
        
            
            
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
              
                metrics = self.graph_discov(trainx, trainy[:,0,:,:],loss_weights)
            
            
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
              
                metrics = self.train_pred(trainx, trainy[:,0,:,:],loss_weights)
                train_loss.append(metrics[0])
                train_mae.append(metrics[1])
                train_mape.append(metrics[2])
                train_rmse.append(metrics[3])
                
            
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mae = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = self.eval(testx, testy[:,0,:,:])
                valid_loss.append(metrics[0])
                valid_mae.append(metrics[1])
                valid_mape.append(metrics[2])
                valid_rmse.append(metrics[3])
                
            s2 = time.time()
            
        
            
            self.graph_scheduler.step()
            
            self.gumbel_tau *= self.gumbel_tau_gamma
            self.lambda_s *= self.lambda_gamma
            
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)
            # t_loss.append(mtrain_loss)
            mvalid_loss = np.mean(valid_loss)
            mvalid_mae = np.mean(valid_mae)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            val_loss.append(mvalid_loss)
            val_mae.append(mvalid_mae)
            if mvalid_loss<min_loss:
                min_loss=mvalid_loss
                min_graph=self.graph
                count=0
            else:
                count += 1

            # early stopping
            if  count == self.args.early_stop_patience:
                print('Traning stop')
            
                break   

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            torch.save(self.model.state_dict(), params_path+"/"+self.args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        
        
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
     
       #testing
        bestid = np.argmin(val_loss)
        self.model.load_state_dict(torch.load(params_path+"/"+self.args.model+"_epoch_"+str(bestid+1)+"_"+str(round(val_loss[bestid],2))+".pth"))
        self.model.eval()
        
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1,3)[:,0,:,:]
        print(realy.shape)
        
        graph=self.sample_bernoulli_time(min_graph)[0]
       
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():
                
                preds,_,A= self.model(testx,graph)
               
                preds=preds.transpose(1,3)
                # preds=preds.narrow(1,1,1)
            outputs.append(preds.squeeze())
            
        #print(outputs.shape)
        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]
        print(yhat.shape)

        print("Training finished")
        print("The valid loss on best model is", str(round(val_loss[bestid],4)))
        
      
        
        
        amae = []
        amape = []
        armse = []
        prediction=yhat
        
    
        for i in range(12):
           
            pred = prediction[:,:,i]
            real = realy[:,:,i]
           
            metrics = metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
       
        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        amae.append(round(np.mean(amae),4))
        amape.append(round(np.mean(amape),4))
        armse.append(round(np.mean(armse),4))
        
        amae=np.array(amae)
        amape=np.array(amape)
        armse=np.array(armse)
        np.savetxt(self.args.save+'/'+'test_mae.txt',amae)
        np.savetxt(self.args.save+'/'+'test_mape.txt',amape)
        np.savetxt(self.args.save+'/'+'test_rmse.txt',armse)
      
    
        torch.save(self.model.state_dict(),params_path+"/"+self.args.model+"_exp"+str(self.args.expid)+"_best_"+str(round(val_loss[bestid],2))+".pth")
        prediction_path=params_path+"/"+self.args.model+"_prediction_results"
        ground_truth=realy.cpu().detach().numpy()
        prediction=prediction.cpu().detach().numpy()
        
        np.savez_compressed(
                os.path.normpath(prediction_path),
                prediction=prediction,
                ground_truth=ground_truth
            )


def main(opt:SSHopt, device="cuda"):
    
    
    loss_weights=[1,0.5]
    
  
    if hasattr(opt,"ssh"):
        ssh_model = Engine(opt.ssh, device=device)
        ssh_model.train(loss_weights)
            
        
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'config.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='1', type=str)
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    device = "cuda"
    seed=1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    main(OmegaConf.load(args.opt), device=device)

    