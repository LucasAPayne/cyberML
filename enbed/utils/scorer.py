import torch

def DistMult_score(s_emb, o_emb, r_emb):
    '''
    DistMult triple score function (Yang et al., 2014).
    '''
    return (s_emb*r_emb*o_emb).sum(-1)

def RESCAL_score(s_emb, o_emb, r_emb):
    '''
    RESCAL triple score function (Nickel et al., 2011).
    '''
    return (s_emb.unsqueeze(-1)*torch.matmul(r_emb, o_emb.unsqueeze(-1))).sum(-2).T[0]
