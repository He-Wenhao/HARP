import os;
import json;
import time
import torch

def _to_gpu_index(device):
    try:
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            if device.startswith('cuda:'):
                return int(device.split(':', 1)[1])
            return 0
        if isinstance(device, torch.device):
            return 0 if device.index is None else int(device.index)
    except:
        return 0
    return 0

def start_iter_nvml(device):
    if device == 'cpu':
        return None
    import pynvml as _nv
    import threading as _th
    state = {
        'running': True,
        'util': [],
        'util_mem': [],
        'mem_used': [],
        'thread': None,
        'handle': None
    }
    _nv.nvmlInit()
    # resolve device index
    idx = _to_gpu_index(device)
    handle = _nv.nvmlDeviceGetHandleByIndex(int(idx))
    state['handle'] = handle
    import time as _t
    def _loop():
        while state['running']:
            util = _nv.nvmlDeviceGetUtilizationRates(state['handle'])
            mem = _nv.nvmlDeviceGetMemoryInfo(state['handle'])
            state['util'].append(int(util.gpu))
            state['util_mem'].append(int(util.memory))
            state['mem_used'].append(int(mem.used // (1024*1024)))
            _t.sleep(0.05)
    t = _th.Thread(target=_loop, daemon=True)
    state['thread'] = t
    t.start()
    return state


def record_iter_nvml(iter_nvml_file, epoch, iter_idx, state):
    state['running'] = False
    if state['thread'] is not None:
        state['thread'].join(timeout=0.5)
    if len(state['util']) == 0:
        return
    # ensure header
    if not os.path.exists(iter_nvml_file):
        with open(iter_nvml_file, 'w') as f:
            f.write('epoch\titer\tavg_util_gpu(%)\tavg_util_mem(%)\tavg_mem_used(MB)\n')
    avg_u = sum(state['util'])/len(state['util'])
    avg_um = sum(state['util_mem'])/len(state['util_mem'])
    avg_mem = sum(state['mem_used'])/len(state['mem_used'])
    with open(iter_nvml_file, 'a') as f:
        f.write(f"{epoch}\t{iter_idx}\t{avg_u:.1f}\t{avg_um:.1f}\t{avg_mem:.0f}\n")


class recorder:

    def __init__(self, OPS, rank=0, path='output', resume=False):
        '''
        if rank == 0:
            if(not os.path.exists(path)):
                os.mkdir(path);

            if(not os.path.exists(path+'/loss')):
                os.mkdir(path+'/loss');

            if(not os.path.exists(path+'/model')):
                os.mkdir(path+'/model');

            if(not os.path.exists(path+'/test')):
                os.mkdir(path+'/test');

            if(not os.path.exists(path+'/inference')):
                os.mkdir(path+'/inference');

        torch.distributed.barrier()
        '''

        self.rank = rank;
        self.loss_file = path + '/loss/loss_'+str(rank)+'.txt';
        self.model_path = path + '/model/';
        self.test_file = path + '/test/test.json';
        self.inference_file = path + '/inference/inference.json';
        self.timer_file = f"{path}/loss/timer.txt"
        self.iter_timer_file = f"{path}/loss/iter_timers.txt"
        self.iter_nvml_file = f"{path}/loss/iter_nvml.txt"
        self._start_time = time.time() if rank == 0 else None
        self.col_width = 15

        self.OPS = OPS;

        timer_mode = 'a' if resume else 'w'
        with open(self.timer_file, timer_mode) as file:
            if not resume:
                file.write('epoch\telapsed(s)\tremaining(s)');
                file.write('\n');

        with open(self.iter_timer_file, timer_mode) as file:
            if not resume:
                file.write('epoch\titer\tdata_s\tfwd_bwd_s\tother_s\ttotal_s\n');

        if not os.path.exists(self.iter_nvml_file):
            with open(self.iter_nvml_file, 'w') as f:
                f.write('epoch\titer\tavg_util_gpu(%)\tavg_util_mem(%)\tavg_mem_used(MB)\n')

    def initialize_loss(self, subfolder, resume=False):
        # insert subfolder to self.loss_file
        file_path = os.path.join(os.path.dirname(self.loss_file), subfolder, os.path.basename(self.loss_file))
        if resume and os.path.exists(file_path):
            return  # append to existing file; header already present
        with open(file_path, 'w') as file:
            # Define column width
            col_width = self.col_width
            file.write(f"{'epoch':<{col_width}}{'total':<{col_width}}")
            for name in self.OPS.keys():
                file.write(f"{name:<{col_width}}")
            file.write('\n')

    def record_loss(self, i, loss, total_loss,subfolder):
        # insert subfolder to self.loss_file
        file_path = os.path.join(os.path.dirname(self.loss_file), subfolder, os.path.basename(self.loss_file))
        with open(file_path, 'a') as file:
            col_width = self.col_width
            file.write(f"{i:<{col_width}d}{total_loss:<{col_width}.6e}")
            for val in loss:
                file.write(f"{val:<{col_width}.6e}")
            file.write('\n')

    def save_model(self, train1, i, Nsave):

        if(i%Nsave == 0 and i>0 and self.rank==0):

            train1.save(self.model_path + str(i)+'_model.pt');
            print('saved model at epoch '+str(i));

    def save_checkpoint(self, train1, optimizer, scheduler, epoch, Nsave):
        if epoch % Nsave == 0 and epoch > 0 and self.rank == 0:
            train1.save_checkpoint(
                self.model_path + 'latest_checkpoint.pt',
                optimizer, scheduler, epoch
            )
            print(f'saved checkpoint at epoch {epoch}')

    def save_test(self, properties):
        
        with open(self.test_file, 'w') as file:
            json.dump(properties, file);
        
        return None;
    
    def save_apply(self, properties):
        
        with open(self.inference_file, 'w') as file:
            json.dump(properties, file);
        
        return None;

    def record_timer(self, i, N):
        if self.rank != 0:
            return

        elapsed = time.time() - self._start_time
        est_total = elapsed / (i + 1) * N
        est_remaining = est_total - elapsed
        
        with open(self.timer_file, 'a') as file:
            file.write(str(i)+f"\t{elapsed:.2f}\t{est_remaining:.2f}")
            
            file.write('\n');

    def record_iter_timers(self, epoch, timers):
        if self.rank != 0:
            return
        # timers: list of tuples (data_s, fwd_bwd_s, other_s, total_s)
        with open(self.iter_timer_file, 'a') as file:
            for it, (data_s, fwd_bwd_s, other_s, total_s) in enumerate(timers):
                file.write(f"{epoch}\t{it}\t{data_s:.6f}\t{fwd_bwd_s:.6f}\t{other_s:.6f}\t{total_s:.6f}\n")


