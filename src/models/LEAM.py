import torch
import torch.nn as nn
import torch.nn.functional as F
class LEAM(nn.Module):
    def __init__(self, text_vocab, label_vocab, vec_dim, label_size, hidden_size, r):
        super(LEAM, self).__init__()
        self.text_embedding = nn.Embedding(len(text_vocab), vec_dim)
        self.text_embedding.weight.data.copy_(text_vocab.vectors)

        self.label_embedding = nn.Embedding(len(label_vocab), vec_dim)
        self.label_embedding.weight.data.copy_(label_vocab.vectors)

        self.con1d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2*r+1, padding=r, padding_mode='reflect')

        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size)
        )

    def forward(self, text, label):
        
        V = self.text_embedding(text)
        C = self.label_embedding(label)
        G = torch.stack([F.cosine_similarity(V, C_i, dim=2) for C_i in C.chunk(C.shape[1], dim=1)], dim=2)
        G = G.permute(0, 2, 1)
        U = self.con1d(G)
        U, _ =torch.max(U, dim=1)
        Beta = F.softmax(U, dim=1).unsqueeze(dim=2)

        z = torch.mul(V, Beta)

        out = self.fc(torch.mean(z, dim=1))

        return out
