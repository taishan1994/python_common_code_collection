class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout1D, self).__init__()
        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, maxlen, 1, embed_size)
        x = x.permute(0, 3, 2, 1)  # (N, embed_size, 1, maxlen)
        x = self.dropout2d(x)  # (N, embed_size, 1, maxlen)
        x = x.permute(0, 3, 2, 1)  # (N, maxlen, 1, embed_size)
        x = x.squeeze(2)  # (N, maxlen, embed_size)

        return x
