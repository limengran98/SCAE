import sys
import torch
from torch import nn
import torch
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor


def to_laplacian(edge_index, num_nodes):
    """
    Make a graph Laplacian term for the GMRF loss.
    """
    if isinstance(edge_index, SparseTensor):
        row = edge_index.storage.row()
        col = edge_index.storage.col()
        edge_index = torch.stack([row, col])
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    size = num_nodes, num_nodes
    return torch.sparse_coo_tensor(edge_index, edge_weight, size=size, device=edge_index.device)


def to_mean_loss(features, laplacian):
    """
    Compute the loss term that compares features of adjacent nodes.
    """
    return torch.bmm(features.t().unsqueeze(1), laplacian.matmul(features).t().unsqueeze(2)).view(-1)




def BernoulliLoss(input, target):
    """
    Run forward propagation.
    """
    BCEloss = nn.BCEWithLogitsLoss(reduction='none')
    pos_ratio = (target > 0).float().mean()
    weight = torch.ones_like(target)
    weight[target > 0] = 1 / (2 * pos_ratio)
    weight[target == 0] = 1 / (2 * (1 - pos_ratio))
    loss = BCEloss(input, target) * weight
    return loss.mean()


class GMRFLoss(nn.Module):
    """
    Implementation of the GMRF loss.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()
        self.cached_adj = None
        self.beta = beta

    def forward(self, features, edge_index):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = edge_index

        num_nodes = features.size(0)
        hidden_dim = features.size(1)
        eye = torch.eye(hidden_dim, device=features.device)
        l1 = (eye + features.t().matmul(features) / self.beta).logdet()
        l2 = to_mean_loss(features, self.cached_adj).sum()
        return (l2 - l1 / 2) / num_nodes

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def category_contrastive_loss(repre, gt, classes, flag_gt):
    """Category-level contrastive loss.

    This function computes loss on the representation corresponding to its groundtruth (repre, gt).  A

    Args:
      repre: [N, D] float tensor.
      gt: [N, 1] float tensor.
      classes:  int tensor.

    Returns:
      loss:  float tensor.
    """

    if flag_gt == True:
        gt = gt - 1

    batch_size = gt.size()[0]
    F_h_h = torch.matmul(repre, repre.t())
    F_hn_hn = torch.diag(F_h_h)
    F_h_h = F_h_h - torch.diag_embed(F_hn_hn)

    label_onehot = torch.nn.functional.one_hot(gt, classes).float()

    label_num = torch.sum(label_onehot, 0, keepdim=True)
    F_h_h_sum = torch.matmul(F_h_h, label_onehot)
    label_num_broadcast = label_num.repeat([gt.size()[0], 1]) - label_onehot
    label_num_broadcast[label_num_broadcast == 0] = 1
    F_h_h_mean = torch.div(F_h_h_sum, label_num_broadcast)
    gt_ = torch.argmax(F_h_h_mean, dim=1)  # gt begin from 0
    F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]
    theta = (gt == gt_).float()
    F_h_hn_mean_ = F_h_h_mean.mul(label_onehot)
    F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1)
    return torch.sum(torch.relu(torch.add(theta, torch.sub(F_h_h_mean_max, F_h_hn_mean))))
