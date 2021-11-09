import torch
from torch import nn

class IntentionLstm(nn.Module):
    def __init__(
        self,
        embedding_size=32,
        hidden_size=32,
        num_layers=1,
        dropout=0.,
        bidirectional=False,
        obs_seq_len=4,
        pred_seq_len=6,
        motion_dim=3,
    ):
        super(IntentionLstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=2*embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            )
        self.spatial_embedding = nn.Linear(motion_dim, embedding_size)
        self.intention_embedding = nn.Linear(1, embedding_size)
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.hidden_to_pos = nn.Linear(self.directions*num_layers*hidden_size, motion_dim)
        self.embedding_size = embedding_size
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
    
    def forward(self, b_xobs, b_yintention, device="cuda:0"):
        
        """
        Forward function.
        inputs:
            - b_xobs: batch of observation. (batch, obs_seq_len, 2)
            - b_yintention: batch of intention. (batch, 1)
            - device
        outputs:
            - b_xpred: (batch, pred_seq_len, 2)
            
        """
        batch_size, _, _ = b_xobs.shape
        b_xobs_offset = (b_xobs[:,1:] - b_xobs[:,:-1]).float() # (N, obs_seq_len-1,2)
        b_xobs_offset_embedding = self.spatial_embedding(b_xobs_offset) # (batch, obs_seq_len, embedding_size)
        b_yintention_embedding = self.intention_embedding(b_yintention.unsqueeze(-1)) # (batch, 1, embedding_size)
        b_obs_embedding = torch.cat((b_xobs_offset_embedding, \
                                   b_yintention_embedding*torch.ones(batch_size, self.obs_seq_len-1, self.embedding_size).to(device)), \
                                   dim=2) # (batch, obs_seq_len-1, 2*embedding_size)
        _, (ht, ct) = self.lstm(b_obs_embedding) # (D∗num_layers, batch, hidden_size)
        b_xpred_tt = b_xobs[:,-1:] # (batch, 1, 2)
        b_xpred = []
        b_xpred_tt_offset = self.hidden_to_pos(ht.permute(1,0,2).reshape(batch_size,1,-1)) # (batch, 1, 2)
        b_xpred_tt = b_xpred_tt + b_xpred_tt_offset
        b_xpred.append(b_xpred_tt)
        for tt in range(1, self.pred_seq_len):
            b_pred_tt_offset_embedding = torch.cat((self.spatial_embedding(b_xpred_tt_offset), b_yintention_embedding), dim=2)
            _, (ht, ct) = self.lstm(b_pred_tt_offset_embedding, (ht, ct))
            b_xpred_tt_offset = self.hidden_to_pos(ht.permute(1,0,2).reshape(batch_size,1,-1)) # (batch, 1, 2)
            b_xpred_tt = b_xpred_tt + b_xpred_tt_offset
            b_xpred.append(b_xpred_tt)
        b_xpred = torch.cat(b_xpred, dim=1)
        return b_xpred
