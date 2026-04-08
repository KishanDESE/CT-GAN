import torch
from preprocess_data import load_preprocessed_data
from conditional_sampler import ConditionalSampler
from generator import Generator

NOISE_DIM = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

X, transformer = load_preprocessed_data()
sampler = ConditionalSampler(X, transformer)

cond_dim = sum(i.output_dim for i in sampler.categorical_info)

G = Generator(
    noise_dim=NOISE_DIM,
    cond_dim=cond_dim,
    transformer=transformer
).to(device)

G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

def generate(n):
    with torch.no_grad():
        c, _, _ = sampler.sample(n)
        c = torch.tensor(c, dtype=torch.float32).to(device)
        z = torch.randn(n, NOISE_DIM, device=device)
        fake = G(z, c).cpu().numpy()

    return transformer.inverse_transform(fake)

df = generate(10000)
df.to_csv("synthetic_adult.csv", index=False)
print(df.head())

