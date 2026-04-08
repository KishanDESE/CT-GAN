import torch
import torch.nn as nn
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, transformer, hidden_dims=(256, 256)):
        super().__init__()

        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.transformer = transformer
        self.output_dim = transformer.output_dim

        dims = [noise_dim + cond_dim] + list(hidden_dims) + [self.output_dim]

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.net(x)

        return self._apply_activations(out)

    def _apply_activations(self, out):
        outputs = []
        start = 0

        for info in self.transformer.column_info:
            end = start + info.output_dim
            chunk = out[:, start:end]

            if info.col_type == "continuous":
                # first value → tanh
                value = torch.tanh(chunk[:, :1])
                # remaining → softmax (mode)
                mode = F.softmax(chunk[:, 1:], dim=1)
                outputs.append(torch.cat([value, mode], dim=1))

            else:
                outputs.append(F.softmax(chunk, dim=1))

            start = end

        return torch.cat(outputs, dim=1)
if __name__ == "__main__":
    from preprocess_data import load_preprocessed_data
    from conditional_sampler import ConditionalSampler

    X, transformer = load_preprocessed_data()
    sampler = ConditionalSampler(X, transformer)

    batch_size = 8
    noise_dim = 128

    c, _, _ = sampler.sample(batch_size)
    z = torch.randn(batch_size, noise_dim)

    G = Generator(
        noise_dim=noise_dim,
        cond_dim=c.shape[1],
        transformer=transformer
    )

    fake = G(z, torch.tensor(c))

    print("Fake shape:", fake.shape)

