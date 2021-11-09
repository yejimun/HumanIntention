import numpy as np
import pickle
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from os.path import join, isdir, isfile, exists
from os import listdir
import re
from src.wlstm.model import WarpLSTM
from src.ilstm.model import IntentionLstm


def load_warplstm_model_by_arguments(args, pkg_path, device):
    logdir = args_to_logdir(args, pkg_path)
    model = load_warplstm_model(
        logdir,
        saved_model_epoch=None,
        num_epochs=args.num_epochs,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_lstms=args.num_lstms,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        end_mask=args.end_mask,
        device=device,
    )
    return model

def args_to_logdir(args, pkg_path):
    writername = 'epoch_'+str(args.num_epochs)+'-num_lstms_'+str(args.num_lstms)+'-lr_'+str(args.lr)
    if args.bidirectional:
        writername = writername + '_bi'
    else:
        writername = writername + '_uni'
    if args.end_mask:
        writername = writername + '_end_mask'
    print('config: ', writername)
    logdir = join(pkg_path, 'results', 'wlstm', 'dataset_full_'+str(args.dataset_ver), writername)
    return logdir

def load_warplstm_model(
    logdir,
    saved_model_epoch,
    num_epochs,
    embedding_size,
    hidden_size,
    num_layers,
    num_lstms,
    dropout,
    bidirectional,
    end_mask,
    device,
):
    if not isdir(logdir):
        raise RuntimeError('The folder '+logdir+' is not found.')
    if saved_model_epoch is None:
        saved_epoch = num_epochs
    else:
        saved_epoch = saved_model_epoch
    for filename in listdir(logdir):
        if isfile(join(logdir, filename)) and re.search('.*epoch_'+str(saved_epoch)+'.pt', filename):
            model_filename = join(logdir, filename)
            model = WarpLSTM(
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_lstms=num_lstms,
                dropout=dropout,
                bidirectional=bidirectional,
                end_mask=end_mask,
            ).to(device)
            checkpoint = torch.load(model_filename, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(model_filename + ' is loaded.')
            return model
    raise RuntimeError('The model is not saved at epoch '+str(saved_epoch)+' in '+logdir)

# ilstm
def load_ilstm_model_by_arguments(args, pkg_path, device):
    model_filepath = args_to_ilstm_filepath(args, pkg_path) # ! TBD
    model = load_ilstm_model(
        model_filepath,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        obs_seq_len=args.obs_seq_len,
        pred_seq_len=args.pred_seq_len,
        motion_dim=args.motion_dim,
        device=device,
    )
    return model

def args_to_ilstm_filepath(args, pkg_path):
    # ! naive version
    model_filename = '211106_slow.pt'
    model_filepath = join(pkg_path, 'results', 'ilstm', model_filename)
    return model_filepath

def load_ilstm_model(
    model_filepath,
    embedding_size,
    hidden_size,
    num_layers,
    dropout,
    bidirectional,
    obs_seq_len,
    pred_seq_len,
    motion_dim,
    device,
):
    if not exists(model_filepath):
        raise RuntimeError('The folder '+model_filepath+' is not found.')
    model = IntentionLstm(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        obs_seq_len=obs_seq_len,
        pred_seq_len=pred_seq_len,
        motion_dim=motion_dim,
    ).to(device)
    model_checkpoint = torch.load(model_filepath, map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    print('Loaded configuration: ', model_filepath)
    return model


##### padding functions START #####
def padding(mini_batch, padding_value=0.):
    """
    inputs:
        - mini_batch: list with length of batch_size.
            - sample: tensor # (time_step, num_channels)
                where time_step is variable, num_channels is fixed (typically 2)
        - padding_values: scalar used for padding.  
    outputs:
        - padded_x: padded batch with uniform time steps. # tensor 
            # (batch_size, longest_time_step, num_channels)
        - x_lens: length of each sample in minibatch. # tensor # (batch_size, )
    """
    x_lens = torch.tensor([len(sample) for sample in mini_batch])
    longest_len = max(x_lens)
    batch_size = len(mini_batch)
    num_channels = mini_batch[0].shape[1]
    padded_x = torch.ones((batch_size, longest_len, num_channels))*padding_value
    for i, sample in enumerate(mini_batch):
        padded_x[i, :len(sample)] = sample
    return padded_x, x_lens

def unpadding(padded_x, x_lens):
    """
    inputs:
        - padded_x: padded batch with uniform time steps. # tensor 
            # (batch_size, longest_time_step, num_channels)
        - x_lens: length of each sample in minibatch. # tensor # (batch_size, )  
    outputs:
        - x: list with length of batch_size.
            - sample: tensor # (time_step, num_channels)
                where time_step is variable, num_channels is fixed (typically 2)
    """
    x = []
    for x_pad, x_len in zip(padded_x, x_lens):
        x.append(x_pad[:x_len])
    return x

def padding_mask(x_lens):
    """
    transform lengths of samples to a binary mask.
    inputs:
        - x_lens: length of each sample in minibatch. # tensor # (batch_size, )  
    outputs:
        - mask: 1-0 binary mask. 1 means valid and 0 means invalid.
            # tensor # (batch_size, longest_time_step, 1)
    """
    longest_len = max(x_lens)
    batch_size = len(x_lens)
    mask = torch.zeros(batch_size, longest_len, 1)
    for i, x_len in enumerate(x_lens):
        mask[i, :x_len] = 1.
    return mask
##### padding functions END #####


##### aoe and moe for variable loss masks START #####
def offset_error(x_gt, x_pred, loss_mask):
    """
    masked offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
          We now want mask like [0, 0, ..., 1, 1, ..,1, 0., 0] to work. And it works. 
    outputs:
        - oe: (batch, seq_len)
    """
    oe = (((x_gt - x_pred) ** 2.).sum(dim=2)) ** (0.5) # (batch, seq_len)
    oe_masked = oe * loss_mask.squeeze(dim=2) # (batch, seq_len)
    return oe_masked

def average_offset_error(x_gt, x_pred, loss_mask):
    """
    average offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
    outputs:
        - aoe: scalar
    """
    oe_masked = offset_error(x_gt, x_pred, loss_mask) # (batch, seq_len)
    aoe = oe_masked.sum(dim=1)/loss_mask.sum(dim=2).sum(dim=1) # (batch) # we assume loss_mask for each sample has at least a valid one.
    aoe = aoe.mean()
    return aoe

def offset_error_square(x_gt, x_pred, loss_mask):
    """
    masked offset error square for global coordinate data (only for training).
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
          We now want mask like [0, 0, ..., 1, 1, ..,1, 0., 0] to work. And it works. 
    outputs:
        - oe: (batch, seq_len)
    """
    oe = (((x_gt - x_pred) ** 2.).sum(dim=2)) # (batch, seq_len)
    oe_masked = oe * loss_mask.squeeze(dim=2) # (batch, seq_len)
    return oe_masked


def average_offset_error_square(x_gt, x_pred, loss_mask):
    """
    average offset error square for global coordinate data. (for training)
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
    outputs:
        - aoe: scalar
    """
    oe_masked = offset_error_square(x_gt, x_pred, loss_mask) # (batch, seq_len)
    aoe = oe_masked.sum(dim=1)/loss_mask.sum(dim=2).sum(dim=1) # (batch) # we assume loss_mask for each sample has at least a valid one.
    aoe = aoe.mean()
    return aoe

def max_offset_error(x_gt, x_pred, loss_mask):
    """
    max offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
    outputs:
        - moe: scalar
    """
    oe_masked = offset_error(x_gt, x_pred, loss_mask) # (batch, seq_len)
    moe, _ = oe_masked.max(dim=1) # (batch)
    moe = moe.mean()
    return moe

def final_offset_error(x_gt, x_pred, loss_mask):
    """
    final offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
    outputs:
        - foe: scalar
    """
    oe_masked = offset_error(x_gt, x_pred, loss_mask) # (batch, seq_len)
    foe_list = []
    for i, loss_mask_row in enumerate(loss_mask[:,:,0]): # find the last valid index of 1 in loss mask for each row
        last_valid_idx = np.where(loss_mask_row.to('cpu')==1)[0][-1]
        foe_list.append(oe_masked[i, last_valid_idx].unsqueeze(0)) # unsqueeze to help do mean in torch
    foe = torch.cat(foe_list, dim=0).mean()
    return foe
##### aoe and moe for variable loss masks END #####


##### load datasets START #####
def load_preprocessed_train_test_dataset(pkg_path, dataset_ver=0):
    """
    Inputs:
        - dataset_ver: int. 0, 25, 50, or 75 to represent the percentage of observation.
    
    Outputs:
        - traj_base_train: list of tensors with shape (time_step, 2)
        - traj_true_train: list of tensors with shape (time_step, 2)
        - traj_loss_mask_train: list of tensors with shape (time_step, 1)
        - traj_base_test: list of tensors with shape (time_step, 2)
        - traj_true_test: list of tensors with shape (time_step, 2)
        - traj_loss_mask_test: list of tensors with shape (time_step, 1)
    """
    ##### Load Dataset #####
    dataset_filename = 'full_'+str(dataset_ver)+'.p'
    dataset_filepath = join(pkg_path, 'datasets', dataset_filename)
    with open(join(dataset_filepath), 'rb') as f:
        x_dict_list_tensor = pickle.load(f)
        print()
        print('LOAD DATASET')
        print(dataset_filename+' is loaded.')
        print()
    traj_base_train, traj_true_train, traj_loss_mask_train, \
        traj_base_test, traj_true_test, traj_loss_mask_test = \
        x_dict_list_tensor['train']['base'], x_dict_list_tensor['train']['true'], x_dict_list_tensor['train']['loss_mask'], \
        x_dict_list_tensor['test']['base'], x_dict_list_tensor['test']['true'], x_dict_list_tensor['test']['loss_mask']
    ##### Transform loss mask data shape #####
    for i in range(len(traj_loss_mask_train)):
        traj_loss_mask_train[i] = traj_loss_mask_train[i].reshape(-1, 1)
    for i in range(len(traj_loss_mask_test)):
        traj_loss_mask_test[i] = traj_loss_mask_test[i].reshape(-1, 1) # loss_mask: (seq_len, 1)
    return traj_base_train, traj_true_train, traj_loss_mask_train, \
        traj_base_test, traj_true_test, traj_loss_mask_test
##### load datasets END #####

##### intention aware linear model START #####
def ilm(zipped_data_train_test):
    """
    inputs:
        - zipped_data_train_test: list.
            - traj_base_train: list of tensors with shape (time_step, 2)
            - traj_true_train: list of tensors with shape (time_step, 2)
            - traj_loss_mask_train: list of tensors with shape (time_step, 1)
            - traj_base_test: list of tensors with shape (time_step, 2)
            - traj_true_test: list of tensors with shape (time_step, 2)
            - traj_loss_mask_test: list of tensors with shape (time_step, 1)
    outputs:
        - print aoe and moe and foe on test dataset by using the intention-aware linear model.
    """
    _, _, _, \
        traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_train_test
    sb, sl = padding(traj_base_test)
    st, _ = padding(traj_true_test)
    sm_pred, _ = padding(traj_loss_mask_test) # remember it is (time_step, 1)
    aoe = average_offset_error(sb, st, sm_pred)
    moe = max_offset_error(sb, st, sm_pred)
    foe = final_offset_error(sb, st, sm_pred)
    print()
    print('BASELINE COMPUTATION')
    print('iLM aoe: {0:.4f} | moe: {1:.4f} | foe: {2:.4f}'.format(aoe, moe, foe))
    print()
    return
##### intention aware linear model END #####


##### batch iteration functions START #####
def batch_iter(*dataset, batch_size=32, drop_last=False):
    """
    iteration to give minibatches of corresponding data.
    inputs:
        - dataset: lists with the same length (same sample_num).
            e.g. traj_base_train, traj_true_train, traj_loss_mask_train
        - batch_size: batch size.
        - drop_last: whether drop the last batch with a batch size smaller than batch_size.
    outputs:
        - mini_batch: one mini batch with batch_size at a time.
            e.g. for samples_base, samples_true, samples_loss_mask in
            iter(traj_base_train, traj_true_train, traj_loss_mask_train, batch_size=4):
    """
    sample_num = len(dataset[0])
    sampler = BatchSampler(SubsetRandomSampler(range(sample_num)), \
                           batch_size, drop_last=drop_last) # can turn drop_last off if wanted
    for indices in sampler:
        # print(indices)
        mini_batch = []
        for data in dataset:
            mini_batch.append([data[i] for i in indices])
        yield mini_batch

def batch_iter_no_shuffle(*dataset, batch_size=32, drop_last=False):
    """
    iteration to give minibatches of corresponding data.
    inputs:
        - dataset: lists with the same length (same sample_num).
            e.g. traj_base_train, traj_true_train, traj_loss_mask_train
        - batch_size: batch size.
        - drop_last: whether drop the last batch with a batch size smaller than batch_size.
    outputs:
        - mini_batch: one mini batch with batch_size at a time.
            e.g. for samples_base, samples_true, samples_loss_mask in
            iter(traj_base_train, traj_true_train, traj_loss_mask_train, batch_size=4):
    """
    count = 0
    while count < len(dataset[0]):
        mini_batch = []
        if count + batch_size <= len(dataset[0]):
            for data in dataset:
                mini_batch.append(data[count:count+batch_size])
            yield mini_batch
        count = count + batch_size
##### batch iteration functions END #####