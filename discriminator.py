import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        pac=10,
        hidden_dims=(256, 256)
    ):
        super().__init__()

        self.pac = pac
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        dim = (input_dim + cond_dim) * pac

        dims = [dim] + list(hidden_dims) + [1]

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x, c):
        """
        x: (batch, input_dim)
        c: (batch, cond_dim)
        """

        batch_size = x.size(0)
        assert batch_size % self.pac == 0, "Batch size must be divisible by pac"

        x = x.view(batch_size // self.pac, -1)
        c = c.view(batch_size // self.pac, -1)

        inp = torch.cat([x, c], dim=1)
        return self.net(inp)
def gradient_penalty(discriminator, real, fake, c, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    alpha = alpha.expand_as(real)

    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates, c)

    grad = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()
if __name__ == "__main__":
    from preprocess_data import load_preprocessed_data
    from conditional_sampler import ConditionalSampler
    from generator import Generator

    device = "cpu"

    X, transformer = load_preprocessed_data()
    sampler = ConditionalSampler(X, transformer)

    batch_size = 20
    pac = 10
    noise_dim = 128

    c, _, _ = sampler.sample(batch_size)
    c = torch.tensor(c, dtype=torch.float32)

    z = torch.randn(batch_size, noise_dim)

    G = Generator(
        noise_dim=noise_dim,
        cond_dim=c.shape[1],
        transformer=transformer
    )

    fake = G(z, c)

    D = Discriminator(
        input_dim=transformer.output_dim,
        cond_dim=c.shape[1],
        pac=pac
    )

    out = D(fake, c)
    print("Discriminator output shape:", out.shape)

