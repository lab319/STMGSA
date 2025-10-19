import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from STMGSA import Mix_STMGSA
from Mix_adj import Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch import nn


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def get_feature(adata, deconvolution=False):
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]
    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a


def train_STMGSA(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STMGSA',
                 gradient_clipping=5., weight_decay=0.0001, verbose=True,
                 random_seed=0, save_loss=False, save_reconstrction=False,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dim_input=3000, dim_output=64, alpha=10, beta=1, gamma=1, lane=1,
                 tau=0.5, num_clusters=7, batch_size=100, deconvolution=False,
                 spatial_hidden_dim=64, spatial_dropout=0.1, spatial_batch_size=128,
                 use_scheduler=False, scheduler_patience=30, scheduler_factor=0.7, lr_min=1e-6):
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Construct two kinds of spatial neighborhood network first!")

    print('Size of Input: ', adata_Vars.shape)
    add_contrastive_label(adata)
    get_feature(adata)

    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    adj1 = adata.obsm['adj1']
    adj2 = adata.obsm['adj2']
    graph_neigh1 = torch.FloatTensor(adata.obsm['graph_neigh1'].copy() + np.eye(adj1.shape[0])).to(device)
    graph_neigh2 = torch.FloatTensor(adata.obsm['graph_neigh2'].copy() + np.eye(adj2.shape[0])).to(device)

    dim_input = features.shape[1]

    adj1 = preprocess_adj(adj1)
    adj1 = torch.FloatTensor(adj1).to(device)
    adj2 = preprocess_adj(adj2)
    adj2 = torch.FloatTensor(adj2).to(device)

    model = Mix_STMGSA(dim_input, dim_output, esparse=False, tau=tau, num_clusters=num_clusters,
                       spatial_hidden_dim=spatial_hidden_dim, spatial_dropout=spatial_dropout,
                       spatial_batch_size=spatial_batch_size).to(device)

    loss_CSL = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor,
            patience=scheduler_patience, min_lr=lr_min, verbose=True
        )

    print('Begin to train ST data with Spatial Attention...')
    model.train()

    best_loss = float('inf')
    best_model_state = None

    loss_history = []

    for epoch in tqdm(range(n_epochs)):
        model.train()

        features_a = permutation(features)
        hiden_feat, emb, ret, ret_a, zs, zs_a = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)

        loss_gf_1 = loss_CSL(ret, label_CSL)
        loss_gf_2 = loss_CSL(ret_a, label_CSL)

        loss_feat = F.mse_loss(features, emb)

        loss_ll = model.stmgsa.cont_ll(zs, zs_a, batch_size=batch_size)

        ret_cf = model.stmgsa.cf_forward(zs, zs_a, cluster_temp=100)
        n_samples = zs.size(0)
        expected_size = n_samples * 2

        lbl_1 = torch.ones(1, n_samples).to(device)
        lbl_2 = torch.zeros(1, n_samples).to(device)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        cont_cf = model.stmgsa.cont_bxent(lbl, ret_cf)

        loss = alpha * loss_feat + beta * (loss_gf_1 + loss_gf_2) + gamma * loss_ll + lane * cont_cf

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if use_scheduler:
            scheduler.step(loss)

        curr_lr = optimizer.param_groups[0]['lr']
        loss_record = {
            'epoch': epoch,
            'total_loss': loss.item(),
            'feat_loss': loss_feat.item(),
            'gf_loss': (loss_gf_1 + loss_gf_2).item(),
            'll_loss': loss_ll.item(),
            'cf_loss': cont_cf.item(),
            'lr': curr_lr
        }
        loss_history.append(loss_record)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"Feat: {loss_feat.item():.4f}, GF: {(loss_gf_1 + loss_gf_2).item():.4f}, "
                  f"LL: {loss_ll.item():.4f}, CF: {cont_cf.item():.4f}, "
                  f"LR: {curr_lr:.6f}")

    print("Optimization finished for ST data!")

    if use_scheduler and best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Loaded best model with loss: {best_loss:.4f}")

    with torch.no_grad():
        model.eval()
        if deconvolution:
            emb_rec = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)[1]
        else:
            emb_rec = model(features, features_a, adj1, adj2, graph_neigh1, graph_neigh2)[1].detach().cpu().numpy()

        adata.obsm[key_added] = emb_rec

        if save_loss:
            adata.uns['STMGSA_loss'] = loss.item() if not use_scheduler else best_loss
            adata.uns['STMGSA_loss_history'] = loss_history

        if save_reconstrction:
            ReX = emb.to('cpu').detach().numpy()
            ReX[ReX < 0] = 0
            adata.layers['STMGSA_ReX'] = ReX

        adata.uns['SpatialAttention_params'] = {
            'spatial_hidden_dim': spatial_hidden_dim,
            'spatial_dropout': spatial_dropout,
            'spatial_batch_size': spatial_batch_size
        }

        return adata