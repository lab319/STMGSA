import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Clusterator(nn.Module):
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout
        self.init = torch.rand(self.K, nout)

    def forward(self, embeds, cluster_temp, num_iter=10):
        mu_init, _ = self.cluster(embeds, self.K, 1, num_iter, cluster_temp, self.init)
        mu, r = self.cluster(embeds, self.K, 1, 1, cluster_temp, mu_init.clone().detach())

        return mu, r

    def cluster(self, data, k, temp, num_iter, cluster_temp, init):
        device = data.device
        mu = init.to(device)
        data = data.to(device)
        cluster_temp = torch.tensor(cluster_temp).to(device)

        data = data / (data.norm(dim=1)[:, None] + 1e-6)

        for t in range(num_iter):
            mu = mu / (mu.norm(dim=1)[:, None] + 1e-6)
            dist = torch.mm(data, mu.transpose(0, 1))
            r = F.softmax(cluster_temp * dist, dim=1)
            cluster_r = r.sum(dim=0)
            cluster_mean = r.t() @ data
            new_mu = torch.diag(1 / cluster_r) @ cluster_mean
            mu = new_mu

        r = F.softmax(cluster_temp * dist, dim=1)
        return mu, r


class Discriminator_cluster(nn.Module):
    def __init__(self, n_in, n_h, n_nb, num_clusters):
        super(Discriminator_cluster, self).__init__()

        self.n_nb = n_nb
        self.n_h = n_h
        self.num_clusters = num_clusters

    def forward(self, c, c2, h_0, h_pl, h_mi, S, s_bias1=None, s_bias2=None):
        batch_size = h_0.size(0)
        c_x = c.expand_as(h_0)

        sc_1 = torch.bmm(h_pl.view(batch_size, 1, self.n_h), c_x.view(batch_size, self.n_h, 1))
        sc_2 = torch.bmm(h_mi.view(batch_size, 1, self.n_h), c_x.view(batch_size, self.n_h, 1))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0).view(1, -1)
        return logits


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SpatialAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=64, dropout=0.1, batch_size=128):
        super(SpatialAttention, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size

        self.feature_transform = nn.Linear(in_features, hidden_dim)

        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(0.2)
        )

        self.post_attention = nn.Linear(hidden_dim, in_features)

        self.layer_norm = nn.LayerNorm(in_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
        nn.init.xavier_uniform_(self.attention_score[0].weight)
        nn.init.zeros_(self.attention_score[0].bias)
        nn.init.xavier_uniform_(self.post_attention.weight)
        nn.init.zeros_(self.post_attention.bias)

    def forward(self, x, adj):
        h = self.feature_transform(x)

        N = x.size(0)

        output = torch.zeros_like(x)

        for i in range(0, N, self.batch_size):
            end_idx = min(i + self.batch_size, N)
            batch_nodes = slice(i, end_idx)

            batch_h = h[batch_nodes]

            batch_adj = adj[batch_nodes]

            attn_scores = self.attention_score(batch_h).expand(-1, N)

            masked_scores = torch.where(batch_adj > 0, attn_scores, torch.tensor(-9e15).to(x.device))

            attention_weights = F.softmax(masked_scores, dim=1)
            attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)

            neighbor_features = torch.matmul(attention_weights, h)

            transformed_features = self.post_attention(neighbor_features)

            output[batch_nodes] = self.layer_norm(x[batch_nodes] + transformed_features)

        return output


class STMGSA(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, esparse=False, tau=0.5, num_clusters=7,
                 spatial_hidden_dim=64, spatial_dropout=0.1, spatial_batch_size=128):
        super(STMGSA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.dropout = dropout
        self.act = act
        self.esparse = esparse
        self.num_clusters = num_clusters

        self.spatial_hidden_dim = spatial_hidden_dim
        self.spatial_dropout = spatial_dropout
        self.spatial_batch_size = spatial_batch_size

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.spatial_attention_encoder = SpatialAttention(
            self.out_features,
            hidden_dim=spatial_hidden_dim,
            dropout=spatial_dropout,
            batch_size=spatial_batch_size
        )

        self.spatial_attention_decoder = SpatialAttention(
            self.in_features,
            hidden_dim=spatial_hidden_dim,
            dropout=spatial_dropout,
            batch_size=spatial_batch_size
        )

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        self.fc1 = nn.Linear(self.out_features, 128)
        self.fc2 = nn.Linear(128, self.out_features)

        self.cluster = Clusterator(self.out_features, num_clusters)
        self.disc_c = Discriminator_cluster(self.out_features, self.out_features,
                                            n_nb=None,
                                            num_clusters=num_clusters)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def batched_semi_loss(self, z1, z2, batch_size):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def cont_ll(self, z1, z2, batch_size=0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        return ret.mean()

    def cf_forward(self, h_1, h_2, cluster_temp):
        Z, S = self.cluster(h_1, cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        c2 = self.sigm(c2)

        self.disc_c.n_nb = h_1.size(0)
        ret = self.disc_c(c2, c2, h_1, h_1, h_2, S, None, None)
        return ret

    def cont_bxent(self, lbl, logits):
        b_xent = nn.BCEWithLogitsLoss()
        cont_bxent = b_xent(logits, lbl)
        return cont_bxent

    def forward(self, feat, feat_a, adj, graph_neigh):
        z = F.dropout(feat, self.dropout, self.training)

        z = torch.mm(z, self.weight1)
        if self.esparse:
            z = torch.spmm(adj, z)
        else:
            z = torch.mm(adj, z)

        z = self.spatial_attention_encoder(z, adj)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        h = self.spatial_attention_decoder(h, adj)

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        if self.esparse:
            z_a = torch.spmm(adj, z_a)
        else:
            z_a = torch.mm(adj, z_a)

        z_a = self.spatial_attention_encoder(z_a, adj)

        emb_a = self.act(z_a)

        g = self.read(emb, graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a, emb, emb_a


class Mix_STMGSA(torch.nn.Module):
    def __init__(self, in_features, out_features, esparse=False, tau=0.5, num_clusters=7,
                 spatial_hidden_dim=64, spatial_dropout=0.1, spatial_batch_size=128):
        super(Mix_STMGSA, self).__init__()

        self.attention1 = Attention(out_features)
        self.attention2 = Attention(in_features)
        self.attention3 = Attention(2)
        self.stmgsa = STMGSA(in_features, out_features, esparse=esparse, tau=tau, num_clusters=num_clusters,
                             spatial_hidden_dim=spatial_hidden_dim, spatial_dropout=spatial_dropout,
                             spatial_batch_size=spatial_batch_size)

    def forward(self, features, features_a, adj1, adj2, graph_neigh1, graph_neigh2):
        hiden_emb1, h1, ret1, ret_a1, emb1, emb_a1 = self.stmgsa(features, features_a, adj1, graph_neigh1)
        hiden_emb2, h2, ret2, ret_a2, emb2, emb_a2 = self.stmgsa(features, features_a, adj2, graph_neigh2)

        hiden_emb = torch.stack([hiden_emb1, hiden_emb2], dim=1)
        h = torch.stack([h1, h2], dim=1)
        ret = torch.stack([ret1, ret2], dim=1)
        ret_a = torch.stack([ret_a1, ret_a2], dim=1)
        emb = torch.stack([emb1, emb2], dim=1)
        emb_a = torch.stack([emb_a1, emb_a2], dim=1)

        hiden_emb, _ = self.attention1(hiden_emb)
        h, _ = self.attention2(h)
        ret, _ = self.attention3(ret)
        ret_a, _ = self.attention3(ret_a)
        emb, _ = self.attention1(emb)
        emb_a, _ = self.attention1(emb_a)

        return hiden_emb, h, ret, ret_a, emb, emb_a