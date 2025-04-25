__all__ = ['TrackTimerCB', 'TrackTrainingCB', 'PrintResultsCB', 'TerminateOnNaNCB',
            'TrackerCB', 'SaveModelCB', 'EarlyStoppingCB', 'CSVLogger']

from ..basics import *
from .core import Callback
import torch
import time
import numpy as np
from pathlib import Path
import csv # 將訓練過程中每一個 epoch 的損失與評估指標儲存為 .csv 檔


class TrackTimerCB(Callback):
    def __init__(self):
        super().__init__()
        

    def before_fit(self):
        self.learner.epoch_time = None

    def before_epoch_train(self):         
        self.start_time = time.time()

    def after_epoch_train(self): 
        self.learner.epoch_time = self.format_time(time.time() - self.start_time)

    def format_time(self, t):
        "Format `t` (in seconds) to (h):mm:ss"
        t = int(t)
        h, m, s = t // 3600, (t // 60) % 60, t % 60
        if h != 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'


class TrackTrainingCB(Callback):

    def __init__(self, train_metrics=False, valid_metrics=True):
        super().__init__()        
        self.train_metrics, self.valid_metrics = train_metrics, valid_metrics 

    def init_cb_(self):
        self.setup()    
        self.initialize_recorder()        
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False   

    def before_fit(self):        
        self.setup()    
        self.initialize_recorder()        
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False        
    
    def setup(self):
        self.valid_loss = False
        if self.learner.dls: 
            if not self.learner.dls.valid: self.valid_metrics = False    
            else: self.valid_loss = True

        if self.metrics:
            if not isinstance(self.metrics, list): self.metrics = [self.metrics]   
            self.metric_names = [func.__name__ for func in self.metrics]                       
        else: self.metrics, self.metric_names = [], []        
            
    def initialize_recorder(self):
        recorder = {'epoch': [],  'train_loss': []} 
        if self.valid_loss: recorder['valid_loss'] = []

        for name in self.metric_names: 
            if self.train_metrics: recorder['train_'+name] = []            
            if self.valid_metrics: recorder['valid_'+name] = []
        self.recorder = recorder        
        self.learner.recorder = recorder            
        

    def initialize_batch_recorder(self, with_metrics):
        batch_recorder = {'n_samples': [], 'batch_losses': [], 'with_metrics': with_metrics}                                                         
        self.batch_recorder = batch_recorder

    def reset(self): 
        self.targs, self.preds = [],[]                
        self.n_samples = 0
        self.batch_loss = []


    def after_epoch(self):
        self.recorder['epoch'].append(self.epoch)
        self.learner.recorder = self.recorder              
        
    def before_epoch_train(self): 
        # define storage for batch training loss and metrics        
        self.initialize_batch_recorder(with_metrics=self.train_metrics)        
        self.reset()

    def before_epoch_valid(self):            
        # if valid data is available, define storage for batch training loss and metrics
        # if self.dls.valid:  self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.reset()


    def after_epoch_train(self):         
        values = self.compute_scores()           
        # save training loss after one epoch                
        self.recorder['train_loss'].append( values['loss'] )
        # save metrics after one epoch         
        if self.train_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['train_'+name].append( values[name] ) 
            

    def after_epoch_valid(self):             
        # if there is no valid data, don't store
        if not self.learner.dls.valid: return
        values = self.compute_scores()                
        # save training loss after one epoch
        self.recorder['valid_loss'].append( values['loss'] )
        # save metrics after one epoch         
        if self.valid_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['valid_'+name].append( values[name] ) 
            
    
    def after_batch_train(self): self.accumulate()  # save batch recorder                
    def after_batch_valid(self): self.accumulate()
        
    def accumulate(self ):
        xb, yb = self.batch
        bs = len(xb)                                
        self.batch_recorder['n_samples'].append(bs)
        # get batch loss 
        loss = self.loss.detach()*bs if self.mean_reduction_ else self.loss.detach()        
        self.batch_recorder['batch_losses'].append(loss)
        
        if yb is None: self.batch_recorder['with_metrics'] = False
        if len(self.metrics) == 0: self.batch_recorder['with_metrics'] = False
        # accumulate prediction and target          
        if self.batch_recorder['with_metrics']:
            self.preds.append(self.pred.detach().cpu())
            self.targs.append(yb.detach().cpu())
    

    def compute_scores(self):
        "calculate losses and metrics after each epoch"
        values = {}
        # calculate loss after each epoch        
        n = sum(self.batch_recorder['n_samples'])   # get total number of samples        
        values['loss'] = sum(self.batch_recorder['batch_losses']).item()/n  # averaging

        # calculate metrics if available after each epoch
        if len(self.preds) == 0: return values
        self.preds = torch.cat(self.preds)
        self.targs = torch.cat(self.targs)        
        for func in self.metrics:             
            # values[func.__name__] = func(self.targs, self.preds)
            values[func.__name__] = func(self.targs, self.preds)        
        return values
    

class TerminateOnNaNCB(Callback):
    " A callback to stop the training if loss is NaN"
    def after_batch_train(self):
        if torch.isinf(self.loss) or torch.isnan(self.loss): raise KeyboardInterrupt


class PrintResultsCB(Callback):
    """
    -- Learner.recorder 儲存每一個 epoch 訓練過程中的記錄,像是: train_loss, valid_loss, mse, mae 等。
    -- get_header()：把 recorder 的 key(指標名稱)抓出來，例如 ['train_loss', 'valid_loss', 'mse', 'mae'] + 'time'
    -- before_fit()：在訓練一開始會印出「表頭」。
    -- after_epoch():每次訓練完一個epoch,取出最新的數值,並使用 self.print_value.format(*epoch_logs) 印出結果。
    """
    def __init__(self):
        super().__init__()        

    def get_header(self, recorder):        
        "recorder is a dictionary"
        header = list(recorder.keys()) # 取出表頭  
        return header+['time'] # 加上一個 'time'

    def before_fit(self):
        if self.run_finder: return # don't print if lr_finder is called。若是 lr_finder() 階段，不顯示。
        if not hasattr(self.learner, 'recorder'): return      # don't print if there is no recorder。若沒有 recorder，不顯示。
        header = self.get_header(self.learner.recorder) # 會從 recorder 裡面抓出紀錄的 key。
        self.print_header = '{:>15s}'*len(header) # 靠右對齊、寬度為 15（不夠就補空格）、這欄是字串（string）。
        self.print_value = '{:>15d}' + '{:>15.6f}'*(len(header)-2) + '{:>15}' # 用來輸出每行「數值」的格式
        print(self.print_header.format(*header))        
    
    def after_epoch(self):      
        if self.run_finder: return # don't print if lr_finder is called。若是學習率尋找模式，不顯示結果。避免在 lr_finder()（尋找學習率階段）時，印出多餘的資訊。
        if not hasattr(self.learner, 'recorder'): return  # don't print if there is no recorder。初始化階段（模型未經訓練），若沒有 recorder，不顯示。
        epoch_logs = []        
        for key in self.learner.recorder: # 讀取 learner.recorder 裡紀錄的數值（如 loss、metrics）
            value=self.learner.recorder[key][-1] if self.learner.recorder[key] else None # 抓取每個紀錄項目中最後一筆（最新的）記錄
            epoch_logs += [value]
        if self.learner.epoch_time: epoch_logs.append(self.learner.epoch_time)
        # print('epoch_logs', epoch_logs)
        print(self.print_value.format(*epoch_logs)) # 第 N 個 epoch（從 0 開始）
                                                    # 訓練集上的 loss（training loss）
                                                    # 驗證集上的 loss（validation loss）
                                                    # 評估指標 1（可能是 MSE）
                                                    # 評估指標 2（可能是 MAE）
                                                    # 此 epoch 訓練花費時間（5 秒）        


class CSVLogger(Callback):
    """
    將每個 epoch 的 loss 與 metrics 儲存為 CSV 檔案
    """
    def __init__(self, save_dir='results', filename="epoch_log.csv"):
        super().__init__()
        self.save_path = Path(save_dir) / filename
        self.fields = None
        self.file = None
        self.writer = None
        self.header_written = False

    def before_fit(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.save_path, mode='w', newline='')
        self.writer = csv.writer(self.file)

    def after_epoch(self):
        if not hasattr(self.learner, 'recorder'): return
        row = [self.epoch]
        for key in self.learner.recorder:
            val = self.learner.recorder[key][-1] if self.learner.recorder[key] else ''
            row.append(val)

        if self.learner.epoch_time:
            row.append(self.learner.epoch_time)

        if not self.header_written:
            header = ['epoch'] + list(self.learner.recorder.keys()) + ['time']
            self.writer.writerow(header)
            self.header_written = True

        self.writer.writerow(row)

    def after_fit(self):
        if self.file:
            self.file.close()


class TrackerCB(Callback):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.):
        super().__init__()
        if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
        if comp == np.less: min_delta *= -1
        self.monitor, self.comp, self.min_delta = monitor, comp, min_delta

    def before_fit(self):
        if self.run_finder: return
        if self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf')
        self.monitor_names = list(self.learner.recorder.keys())
        assert self.monitor in self.monitor_names

    def after_epoch(self):        
        if self.run_finder: return
        val = self.learner.recorder[self.monitor][-1]
        if self.comp(val - self.min_delta, self.best): self.best, self.new_best = val,True
        else: self.new_best = False


class SaveModelCB(TrackerCB):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0., 
                        every_epoch=False, fname='model', path=None, with_opt=False, save_process_id=0, global_rank=None):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)        
        self.every_epoch = every_epoch
        self.last_saved_path = None
        self.path, self.fname = path, fname
        self.with_opt = with_opt
        self.save_process_id = save_process_id

        # Identify the worker that saves the model to a file: check if the process' global_rank == save_process_id
        # If running locally using either a cpu/gpu without using DDP -> set save_process_id = global_rank
        # Else if running in DDP mode but user doesn't specify global_rank -> global_rank = current_device
        #       (local_rank 0 from each node will save the model)
        # Else if user provides the global_rank -> use the global_rank to check

        if global_rank:
            self.global_rank = int(global_rank)
        else:
            if torch.cuda.is_available():
                self.global_rank = torch.cuda.current_device()
                if not torch.distributed.is_initialized():
                    self.save_process_id = self.global_rank
            else:
                self.global_rank = 0


    def _save(self, fname, path):
        if self.global_rank == self.save_process_id:
            self.last_saved_path = self.learner.save(fname, path, with_opt=self.with_opt)

    def after_epoch(self):
        if self.every_epoch:
            if ((self.epoch%self.every_epoch) == 0) or (self.epoch==self.n_epochs-1): 
                self._save(f'{self.fname}_{self.epoch}', self.path)                            
        else:
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save(f'{self.fname}', self.path)

    def after_fit(self):
        if self.run_finder: return
        if not self.every_epoch and self.global_rank == self.save_process_id:
            self.learner.load(self.last_saved_path, with_opt=self.with_opt)


class EarlyStoppingCB(TrackerCB):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0,
                        patient=5):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.patient = patient
    
    def before_fit(self):
        # set the impatient level
        self.impatient_level = 0
        super().before_fit()
    
    def after_epoch(self):
        super().after_epoch()
        if self.new_best: self.impatient_level = 0   # reset the impatience
        else:
            self.impatient_level += 1
            if self.impatient_level > self.patient:
                print(f'No improvement since epoch {self.epoch-self.impatient_level}: early stopping')
                raise KeyboardInterrupt



