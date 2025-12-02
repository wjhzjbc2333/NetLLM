import os
import sys
import numpy as np
import torch
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.low_rank import peft_model
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm_llama
from plm_special.utils.console_logger import ConsoleLogger
from ABRLLM_v2 import ABRLLM

def save_model(args, model, save_dir):
    if args.rank > 0:
        # save lora weights
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # lora is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load lora weights
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # lora is disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.

    total_train_losses = []
    for epoch in range(args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        #dont save at same time
        if (epoch+1) % args.save_checkpoint_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, str(epoch))
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_model(args, model, checkpoint_dir_epoch)
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        #dont save at previous 10 steps
        if epoch > 10 and epoch % args.eval_per_epoch == 0:
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
                                        process_reward_fn=eval_process_reward_fn)
            episodes_return = eval_logs['episodes_return']
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                save_model(args, model, best_model_dir)
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn):
    model = load_model(args, model, model_dir)
    print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print(results)
    print('Test time:', results['time'])
    print('QoE Metrics:')
    print('  Mean QoE (per chunk):', results['mean_qoe'])
    print('  Total QoE (all episodes):', results['total_qoe'])
    print('  Episodes count:', results['episodes_count'])
    print('  Total chunks:', results['total_chunks'])
    print('  Mean reward (backward compatibility):', results['mean_reward'])
    print('Results saved at:', result_dir)


def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create environment setting
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    args.trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        args.trace_num = len(all_file_names)
    if args.trace_num == len(all_file_names):
        args.fixed_order = True

    env_settings = {
        'all_cooked_time': all_cooked_time,
        'all_cooked_bw': all_cooked_bw,
        'all_file_names': all_file_names,
        'all_mahimahi_ptrs': all_mahimahi_ptrs,
        'video_size_dir': video_size_dir,
        'fixed': args.fixed_order,
        'trace_num': args.trace_num,
    }

    # 3. create training dataset, fetch info
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 4. create ABRLLM model
    # ABRLLM will load PLM and create state encoder internally
    
    # 4.1 Prepare model path
    model_path = os.path.join(cfg.plm_dir, args.plm_type, args.plm_size)
    args.model_path = model_path
    
    # 4.2 Set llm_dim from config (required by ABRLLM)
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    args.llm_dim = plm_embed_size
    
    # 4.3 Create ABRLLM model
    # ABRLLM will load PLM and create state encoder internally
    abrllm_model = ABRLLM(args)
    
    # Ensure model is on the correct device
    abrllm_model.device = torch.device(args.device)
    abrllm_model = abrllm_model.to(args.device)
    
    # 4.5 Apply LoRA if needed (after model creation, modify the plm)
    if args.rank != -1:
        abrllm_model.plm = peft_model(abrllm_model.plm, args.plm_type, rank=args.rank)
    
    # 5. handling directory and path
    # extract training experience pool information
    train_exp_pool_info = args.exp_pool_path.split('/')[-4:-1]
    train_exp_pool_info = '_'.join(train_exp_pool_info)
    
    # Build model directory name with ABRLLM specific parameters
    models_dir = os.path.join(
        cfg.plm_ft_dir, 
        f'{args.plm_type}_{args.plm_size}', 
        train_exp_pool_info + f'_ss_{args.sample_step}', 
        f'abrllm_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'
        f'_sattn_{args.state_use_self_attention}_sahd_{args.state_attn_hidden_dim}_fusion_{args.fusion_method}'
        f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}'
    )
    results_dir = os.path.join(
        cfg.results_dir, 
        f'{args.trace}_{args.video}', 
        f'trace_num_{args.trace_num}_fixed_{args.fixed_order}', 
        f'{args.plm_type}_{args.plm_size}',
        f'abrllm_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}'
    )
    checkpoint_dir = os.path.join(models_dir, 'checkpoint')
    best_model_dir = os.path.join(models_dir, 'best_model')

    # 6. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.adapt:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        console_log = open(os.path.join(models_dir, 'console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        adapt(args, abrllm_model, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, process_reward)
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.model_dir if args.model_dir is not None else best_model_dir
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, abrllm_model, exp_dataset_info, env_settings, model_dir, results_dir, process_reward)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # training dataset settings
    parser.add_argument('--exp-pool-path', help='the path storing the experience pool file for training', default='artifacts/exp_pools/exp_pool.pkl')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    
    # environment settings
    parser.add_argument('--trace', help='name of traces (e.g., fcc-test)', type=str, default='fcc-test')
    parser.add_argument('--trace-num', help='number of traces. if set to -1, use all traces in the trace dir.', type=int, default=100)
    parser.add_argument('--video', help='name of video (e.g., video1)', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    
    # plm settings
    parser.add_argument('--plm-type', type=str, default='llama', help='type of PLM (e.g., llama)')
    parser.add_argument('--plm-size', type=str, default='base', help='size of PLM (e.g., base)')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=-1)
    
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    parser.add_argument('--state-embedding-dim', type=int, help='embedding dim for state encoder (defaults to state-feature-dim)', default=None)
    
    # ABRLLM specific settings
    parser.add_argument('--frozen', action='store_true', help='freeze LLM parameters')
    parser.add_argument('--num-heads', type=int, help='number of attention heads', default=8)
    parser.add_argument('--key-dim', type=int, help='key dimension for alignment layer', default=128)
    parser.add_argument('--state-use-self-attention', action='store_true', help='use self-attention for state features (default: True, set via --no-state-use-self-attention to disable)')
    parser.add_argument('--state-attn-hidden-dim', type=int, help='hidden dimension for state self-attention fusion (defaults to 6 * state-embedding-dim)', default=None)
    parser.add_argument('--fusion-method', type=str, choices=['weighted_sum', 'mean', 'concat'], default='weighted_sum', help='fusion method for state features')
    
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=20)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='warmup steps for learning rate scheduler')
    parser.add_argument('--num-epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--eval-per-epoch', type=int, help='evaluation per epoch', default=1)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='saving checkpoint per iteration', default=10)
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32, help='gradient accumulation steps')
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--model-dir', help='model weight dir for testing', default=None)
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment', default=None)
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output', default=None)
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output', default=None)
    
    # Set default values for boolean flags (before parsing)
    parser.set_defaults(frozen=True)
    parser.set_defaults(state_use_self_attention=True)
    
    args = parser.parse_args()
    
    if args.state_embedding_dim is None:
        args.state_embedding_dim = args.state_feature_dim
    
    #6 * args.state_embedding_dim in ABRLLM, args.state_embedding_dim in ABRLLM_v2
    if args.state_attn_hidden_dim is None:
        args.state_attn_hidden_dim = args.state_embedding_dim
    
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if args.device_out is None:
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    
    assert args.save_checkpoint_per_epoch <= args.num_epochs, 'save_checkpoint_per_epoch should not exceed num_epochs'
    
    # Set max_length for ABRLLM (used for history deque)
    args.max_length = args.w
    
    print('Arguments:')
    pprint(args)
    
    run(args)

