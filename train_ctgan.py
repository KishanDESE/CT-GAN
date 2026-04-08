import torch
import torch.optim as optim
import numpy as np

from preprocess_data import load_preprocessed_data
from conditional_sampler import ConditionalSampler, cond_loss
from generator import Generator
from discriminator import Discriminator, gradient_penalty


BATCH_SIZE = 500
NOISE_DIM = 128
PAC = 10
LAMBDA_GP = 10
LR = 2e-4
BETAS = (0.5, 0.9)
N_CRITIC = 5
EPOCHS = 300


device = "cuda" if torch.cuda.is_available() else "cpu"


X, transformer = load_preprocessed_data()
X = torch.tensor(X, dtype=torch.float32).to(device)

sampler = ConditionalSampler(X.cpu().numpy(), transformer)

cond_dim = sum(i.output_dim for i in sampler.categorical_info)


G = Generator(
    noise_dim=NOISE_DIM,
    cond_dim=cond_dim,
    transformer=transformer
).to(device)

D = Discriminator(
    input_dim=transformer.output_dim,
    cond_dim=cond_dim,
    pac=PAC
).to(device)


opt_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
opt_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)


num_samples = X.size(0)


for epoch in range(EPOCHS):

    for i in range(0, num_samples, BATCH_SIZE):

        if i + BATCH_SIZE > num_samples:
            break

        for _ in range(N_CRITIC):

            cond_vec, mask, col_idx, categories = sampler.sample(BATCH_SIZE)

            real_np = sampler.sample_data(BATCH_SIZE, col_idx, categories)

            real = torch.tensor(real_np, dtype=torch.float32).to(device)
            cond_vec = torch.tensor(cond_vec, dtype=torch.float32).to(device)

            z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)

            fake = G(z, cond_vec).detach()

            d_real = D(real, cond_vec).mean()
            d_fake = D(fake, cond_vec).mean()

            gp = gradient_penalty(D, real, fake, cond_vec, device)

            loss_D = d_fake - d_real + LAMBDA_GP * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        cond_vec, mask, col_idx, categories = sampler.sample(BATCH_SIZE)

        cond_vec = torch.tensor(cond_vec, dtype=torch.float32).to(device)

        z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)

        fake = G(z, cond_vec)

        y_fake = D(fake, cond_vec)

        c_loss = cond_loss(fake, cond_vec, transformer)

        loss_G = -y_fake.mean() + c_loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"D_loss: {loss_D.item():.4f} | "
        f"G_loss: {loss_G.item():.4f}"
    )


def generate_synthetic(n_samples):

    G.eval()

    with torch.no_grad():

        cond_vec, _, _, _ = sampler.sample(n_samples)
        cond_vec = torch.tensor(cond_vec, dtype=torch.float32).to(device)

        z = torch.randn(n_samples, NOISE_DIM, device=device)

        fake = G(z, cond_vec).cpu().numpy()

    df_synth = transformer.inverse_transform(fake)

    return df_synth


df_synthetic = generate_synthetic(10000)

df_synthetic.to_csv("synthetic_adult.csv", index=False)

print(df_synthetic.head())
print("Saved synthetic_adult.csv")

torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

print("Models saved: generator.pth, discriminator.pth")