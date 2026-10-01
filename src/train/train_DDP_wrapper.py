import torch;
import json;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;
import os;

import torch.distributed as dist;
from torch.nn.parallel import DistributedDataParallel as DDP;
import torch.multiprocessing as mp

from train import trainer, recorder;
from data import dataloader;
from model import V_theta;
from util_basis import read_basis;
import pdb
from model.xyz2graph import sampler;


def _gpu_mb(device):
    """Return currently allocated GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / 1e6


def _print_param_report(model, rank):
    """Print total parameter count and per-block breakdown (rank-0 only)."""
    if rank != 0:
        return
    core  = model.module if hasattr(model, 'module') else model
    total = sum(p.numel() for p in core.parameters())

    def _count(m):
        return sum(p.numel() for p in m.parameters())

    print("\n" + "=" * 70)
    print(f"  MODEL PARAMETER REPORT  (Total: {total:,} params, "
          f"~{total * 4 / 1e6:.1f} MB fp32)")
    print("=" * 70)

    if hasattr(core, 'backbone'):
        # DirectPropertyModel path (new backbone abstraction)
        sections = [
            ("backbone  (GNN message passing)", core.backbone),
            ("head      (property prediction)", core.head),
        ]
        # Drill into backbone sub-blocks if InternalBackbone
        bb = core.backbone
        if hasattr(bb, 'embR_func'):
            sections += [
                ("  backbone.embR_func   (radial basis)", bb.embR_func),
                ("  backbone.ele_embed   (element embed)", bb.ele_embed),
            ]
            for idx in range(len(bb.node_update_blocks)):
                sections.append((f"  backbone.node_update_block[{idx}]", bb.node_update_blocks[idx]))
                sections.append((f"  backbone.edge_update_block[{idx}]", bb.edge_update_blocks[idx]))
    else:
        # Legacy V_theta path
        sections = [
            ("embR_func   (Gaussian radial basis)", core.embR_func),
            ("ele_embed   (element embedding MLP)",  core.ele_embed),
        ]
        for idx in range(len(core.node_update_blocks)):
            sections.append((f"node_update_block[{idx}]", core.node_update_blocks[idx]))
            sections.append((f"edge_update_block[{idx}]", core.edge_update_blocks[idx]))
        sections += [
            ("node_E      (per-atom energy head)",      core.node_E),
            ("node_Egap   (per-atom gap head)",         core.node_Egap),
            ("node_charge (per-atom charge head)",      core.node_charge),
            ("node_dipole (per-atom l=1o head)",
                torch.nn.ModuleList([core.node_dipole1, core.node_dipole2])),
            ("node_quad   (per-atom l=0e+2e quad)",
                torch.nn.ModuleList([core.node_quad1, core.node_quad2])),
            ("node_alpha  (per-atom l=0e+2e alpha)",
                torch.nn.ModuleList([core.node_alpha1, core.node_alpha2])),
            ("edge_bond   (per-edge bond order)",
                torch.nn.ModuleList([core.edge_bond1, core.edge_bond2])),
        ]

    for name, mod in sections:
        n   = _count(mod)
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {name:<48s}  {n:>10,}  ({pct:5.1f}%)")
    print("=" * 70 + "\n")


def _print_memory_report(rank, label, device, baseline_mb, model_mb, data_mb):
    """Print a GPU memory breakdown. Pass None for unknown segments."""
    if rank != 0:
        return
    cur = _gpu_mb(device)
    other = max(cur - baseline_mb - model_mb - (data_mb or 0.0), 0.0)
    print(f"\n[GPU Memory @ {label}]  total-allocated={cur:.1f} MB")
    print(f"  model params   : {model_mb:.1f} MB")
    if data_mb is not None:
        print(f"  data (on GPU)  : {data_mb:.1f} MB")
        print(f"  runtime/other  : {other:.1f} MB")
    print("")


def _print_preload_analysis(rank, data_mb, val_data_mb, total_gpu_mb):
    """Print a short analysis of the pre-loading strategy."""
    if rank != 0:
        return
    total_data_mb = data_mb + val_data_mb
    pct = 100.0 * total_data_mb / total_gpu_mb if total_gpu_mb > 0 else 0.0
    print("=" * 70)
    print("  PRE-LOADING STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"  Train data pre-loaded to GPU : {data_mb:.1f} MB")
    print(f"  Val   data pre-loaded to GPU : {val_data_mb:.1f} MB")
    print(f"  Total data on GPU            : {total_data_mb:.1f} MB  ({pct:.1f}% of {total_gpu_mb:.0f} MB)")
    print("")
    print("  Pre-loading verdict:")
    if total_data_mb < 0.3 * total_gpu_mb:
        print("  ✓ GOOD – data footprint is modest relative to GPU capacity.")
        print("    Eliminates all CPU→GPU transfer latency during training,")
        print("    so data sampling is essentially free (see data_s in timers).")
    elif total_data_mb < 0.6 * total_gpu_mb:
        print("  ~ MARGINAL – data consumes a significant fraction of GPU memory.")
        print("    Consider keeping data on CPU and streaming if OOM occurs.")
    else:
        print("  ✗ RISKY – data dominates GPU memory; activations+optimizer states")
        print("    may cause OOM. Consider CPU storage with async prefetch.")
    print("=" * 70 + "\n")

############## Setting parameters ################

def train_func(rank, params):

    OPS = params['OPS'];
    device = params['device'];
    #batch_size = params['batch_size'];
    N_batches = params['N_batches'];
    steps_per_epoch = params['steps_per_epoch'];
    N_epoch = params['N_epoch'];
    lr_init = params['lr_init'];
    lr_final = params['lr_final'];
    lr_decay_steps = params['lr_decay_steps'];
    scaling = params['scaling'];
    Nsave = params['Nsave'];

    element_list = params['element_list'];
    path = params['path'];
    output_path = params['output_path'];
    datagroup = params['datagroup'];
    val_datagroup = params['val_datagroup'];
    load_model = params['load_model'];
    load_model_path = params['load_model_path'];

    world_size = params['world_size'];
    nodeRDM_flag = params['nodeRDM_flag']
    backbone_type   = params.get('backbone_type', 'internal')
    backbone_config = params.get('backbone_config', None)
    delta_learning  = params.get('delta_learning', True)
    # Starting point for delta-learning: 'b3lyp' (default), 'bp86', or 'none'.
    # When delta_learning is False the loader forces baseline='none' internally.
    delta_baseline  = params.get('delta_baseline', 'b3lyp')
    if isinstance(delta_baseline, str):
        delta_baseline = delta_baseline.strip()
    loss_dropout_config = params.get('loss_dropout_config', None)
    # E_gap aggregation: 'sum' (legacy, extensive) or 'mean' (v8+, intensive).
    # If absent in config, fall back to the legacy 'sum' to preserve v7_bp86
    # behaviour when loading older checkpoints.
    egap_pooling = params.get('egap_pooling', 'sum')
    if isinstance(egap_pooling, str):
        egap_pooling = egap_pooling.strip().lower()
    # alpha head: 'sum' (legacy local sum + DFT baseline) or 'thole' (Applequist/
    # Thole relay matrix, predicts full alpha, no baseline; size-extensive).
    alpha_head = params.get('alpha_head', 'sum')
    if isinstance(alpha_head, str):
        alpha_head = alpha_head.strip().lower()
    # Prediction head: 'direct' (per-property heads) or 'hamiltonian' (predict a
    # Fock correction, diagonalize, read observables off the eigenstates). The
    # Hamiltonian path needs the AO operator/charge matrices, so force the loader
    # to keep them (load_obs=True) and pass them into run_model.
    head_type = params.get('head_type', 'direct')
    if isinstance(head_type, str):
        head_type = head_type.strip().lower()
    ham_mode = (head_type == 'hamiltonian')
    # Optional cap on #molecules per rank (handy for single-GPU smoke tests).
    max_items = params.get('max_items', None)
    if isinstance(max_items, str):
        max_items = int(max_items) if max_items.strip() else None
    ###################### load data and model ######################

    basis_path = os.path.join(os.path.dirname(__file__), '..', 'util_basis');
    data_path = path + 'dataset';
    read_basis(path = basis_path, savefile=True);
    print('device:',device)

    # ── GPU memory baseline (before anything is allocated) ──────────────────
    if torch.cuda.is_available():
        torch.cuda.set_device(device)   # initialize CUDA context before any memory queries
        torch.cuda.reset_peak_memory_stats(device)
    baseline_mb = _gpu_mb(device)
    total_gpu_mb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device if isinstance(device, int) else 0)
        total_gpu_mb = props.total_memory / 1e6
    # ────────────────────────────────────────────────────────────────────────

    # load training data
    loader = dataloader(device=device, element_list = element_list,
                            path = data_path,
                            starting_basis = 'def2-SVP');
    # Source of the per-molecule basic features + preprocessing cache. For the
    # B3LYP-baseline experiment set basic_subdir=basic_B3LYP, tmp_subdir=tmp_basic_b3lyp.
    loader.basic_subdir = str(params.get('basic_subdir', 'basic')).strip()
    loader.tmp_subdir = str(params.get('tmp_subdir', 'tmp')).strip()
    if rank == 0:
        print(f"[config] basic_subdir = {loader.basic_subdir}  tmp_subdir = {loader.tmp_subdir}")

    data, labels, obs_mats,charge_mats,  _ = loader.load_data(
        datagroup, rank, world_size,
        delta_learning=delta_learning, delta_baseline=delta_baseline,
        load_obs=ham_mode, max_items=max_items);
    train_sampler = sampler(data, labels, device)
    after_train_data_mb = _gpu_mb(device)
    train_data_mb = max(after_train_data_mb - baseline_mb, 0.0)

    # load validation data
    val_data, val_labels, val_obs_mats,val_charge_mats,  _ = loader.load_data(
        val_datagroup, rank, world_size,
        delta_learning=delta_learning, delta_baseline=delta_baseline,
        load_obs=ham_mode, max_items=max_items);
    val_sampler = sampler(val_data, val_labels, device)
    after_val_data_mb = _gpu_mb(device)
    val_data_mb = max(after_val_data_mb - after_train_data_mb, 0.0)

    train1 = trainer(device, nodeRDM_flag=nodeRDM_flag);

    #train1.build_irreps(element_list = element_list);

    if(world_size>1):
        train1.build_ddp_model(scaling=scaling, element_list=element_list,
                               backbone_type=backbone_type, backbone_config=backbone_config,
                               alpha_head=alpha_head, head_type=head_type)
    else:
        train1.build_model(scaling=scaling, element_list=element_list,
                           backbone_type=backbone_type, backbone_config=backbone_config,
                           alpha_head=alpha_head, head_type=head_type)
    if loss_dropout_config is not None:
        train1.configure_loss_dropout(**loss_dropout_config)
    train1.configure_egap_pooling(egap_pooling)
    gap_rescale = params.get('gap_rescale', 'affine')
    if isinstance(gap_rescale, str):
        gap_rescale = gap_rescale.strip().lower()
    train1.configure_gap_rescale(gap_rescale)
    if rank == 0:
        print(f"[config] egap_pooling = {egap_pooling}")
        print(f"[config] gap_rescale = {gap_rescale}")
        print(f"[config] alpha_head = {alpha_head}")

    # ── Parameter report + memory after model build ──────────────────────────
    _print_param_report(train1.model, rank)
    after_model_mb = _gpu_mb(device)
    model_mb = max(after_model_mb - after_val_data_mb, 0.0)
    if rank == 0:
        print(f"[Rank {rank}] GPU memory snapshot after model build:")
        print(f"  baseline      : {baseline_mb:.1f} MB")
        print(f"  train data    : {train_data_mb:.1f} MB")
        print(f"  val data      : {val_data_mb:.1f} MB")
        print(f"  model params  : {model_mb:.1f} MB")
        print(f"  total GPU     : {total_gpu_mb:.0f} MB (device capacity)\n")
    _print_preload_analysis(rank, train_data_mb, val_data_mb, total_gpu_mb)
    # ────────────────────────────────────────────────────────────────────────
    
    train1.build_optimizer(lr=lr_init);
    #train1.build_charge_matrices(data);

    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                        gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

    start_epoch = 0
    if load_model:
        opt_state, sched_state, start_epoch = train1.load_checkpoint(load_model_path)
        if opt_state is not None:
            train1.optim.load_state_dict(opt_state)
            scheduler.load_state_dict(sched_state)

    # export output dir for per-iteration NVML logs
    os.environ['OUTPUT_DIR'] = output_path

    resume = (start_epoch > 0)
    rec = recorder(OPS, rank=rank, path=output_path, resume=resume);
    rec.initialize_loss(subfolder='train', resume=resume);
    rec.initialize_loss(subfolder='validation', resume=resume);
    ############### Implement model training #####################
    _first_epoch_done = False
    for i in range(start_epoch, N_epoch):
        # set epoch for iter_nvml logs
        os.environ['EPOCH'] = str(i)
        #if rank == 0:
            #print(torch.matmul(alpha_0,T_mats[i]))
        #    pdb.set_trace()

        loss, l_total = train1.run_model(train_sampler,steps=steps_per_epoch,
                            N_batches = N_batches,
                            op_names=OPS,
                            op_matrices=obs_mats, charge_matrices=charge_mats,
                            do_backward=True,time_stat=True);

        # ── After first forward+backward: report runtime GPU memory ─────────
        if not _first_epoch_done and rank == 0:
            _first_epoch_done = True
            cur_mb = _gpu_mb(device)
            # peak = highest allocation seen during the epoch (includes all live intermediates)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else cur_mb
            persistent_mb = max(cur_mb - baseline_mb - train_data_mb - val_data_mb - model_mb, 0.0)
            # peak runtime = peak minus the permanently-resident data+model
            peak_runtime_mb = max(peak_mb - baseline_mb - train_data_mb - val_data_mb - model_mb, 0.0)
            total_alloc = cur_mb - baseline_mb
            print("\n" + "=" * 70)
            print("  GPU MEMORY BREAKDOWN (after first training epoch)")
            print("=" * 70)
            print(f"  model params+buffers      : {model_mb:.1f} MB")
            print(f"  train data (GPU)          : {train_data_mb:.1f} MB")
            print(f"  val   data (GPU)          : {val_data_mb:.1f} MB")
            print(f"  persistent runtime        : {persistent_mb:.1f} MB")
            print(f"   (grads + Adam m/v states after epoch)")
            print(f"  peak runtime during fwd/bwd: {peak_runtime_mb:.1f} MB")
            print(f"   (intermediates: activations, Fock/eigen tensors, autograd graph)")
            print(f"  ─────────────────────────────────────────────────")
            print(f"  current allocated         : {total_alloc:.1f} MB")
            print(f"  peak allocated (epoch 0)  : {peak_mb - baseline_mb:.1f} MB  /  {total_gpu_mb:.0f} MB device")
            print("=" * 70 + "\n")
        # ────────────────────────────────────────────────────────────────────

        # log per-iteration timers for this epoch
        timers = train1.pop_iter_timers();
        rec.record_iter_timers(i, timers);
        scheduler.step();
        rec.record_loss(i, loss, l_total,subfolder=f'train');
        rec.record_timer(i,N_epoch)
        rec.save_model(train1, i, Nsave);
        rec.save_checkpoint(train1, train1.optim, scheduler, i, Nsave);

        val_loss, val_l_total = train1.run_model(val_sampler,
                            steps=steps_per_epoch,
                            N_batches = 1,
                            op_names=OPS,
                            op_matrices=val_obs_mats, charge_matrices=val_charge_mats,
                            do_backward=False,time_stat=True);
        rec.record_loss(i, val_loss, val_l_total,subfolder=f'validation',);
        #raise NotImplementedError("add validation here.")

def train_spawn(rank, params):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=params['world_size'])
    train_func(rank, params);
    dist.destroy_process_group()

    return None;

def train_serial(params):

    train_func(0, params);

    return None;

def train_elastic(params):

    # DDP backend: 'nccl' (fast GPU collectives, default) or 'gloo' (legacy).
    # For nccl each process must own a unique CUDA device BEFORE init, set from
    # torchrun's LOCAL_RANK. device == local_rank (4 GPUs/node => rank%4).
    backend = str(params.get('ddp_backend', 'nccl')).strip().lower()
    local_rank = int(os.environ.get('LOCAL_RANK',
                                    os.environ.get('SLURM_LOCALID', 0)))
    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend);
    rank = dist.get_rank();
    device = local_rank if torch.cuda.is_available() else 'cpu';
    params['device'] = device;

    train_func(rank, params);

    dist.destroy_process_group()

    return None;

def main(params):

    if(params['world_size'] == 1):
        train_serial(params);

    elif(params['ddp_mode'] == 'spawn'):
        mp.spawn(train_spawn,
            args=(params,),
            nprocs=params['world_size'],
            join=True);

    elif(params['ddp_mode'] == 'elastic'):

        train_elastic(params);
    
    else:
        raise ValueError('Invalid ddp_mode');
    
    return None;
