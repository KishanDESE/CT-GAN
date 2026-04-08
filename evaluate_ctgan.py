import pandas as pd
import numpy as np
from scipy.stats import entropy

# -----------------------------
# LOAD DATA
# -----------------------------
real_data = pd.read_csv("adult/adult.data")     # adjust if needed
fake_data = pd.read_csv("synthetic_adult.csv")

# -----------------------------
# KEEP ONLY NUMERIC COLUMNS
# -----------------------------
real_data = real_data.select_dtypes(include=[np.number])
fake_data = fake_data.select_dtypes(include=[np.number])

# -----------------------------
# MATCH SHAPES
# -----------------------------
min_len = min(len(real_data), len(fake_data))
real_data = real_data.iloc[:min_len]
fake_data = fake_data.iloc[:min_len]

real_flat = real_data.values.flatten()
fake_flat = fake_data.values.flatten()

# -----------------------------
# KL DIVERGENCE
# -----------------------------
def kl_divergence(real, fake):
    hist_real, _ = np.histogram(real, bins=50, density=True)
    hist_fake, _ = np.histogram(fake, bins=50, density=True)

    hist_real += 1e-8
    hist_fake += 1e-8

    return entropy(hist_real, hist_fake)

# -----------------------------
# JS DIVERGENCE
# -----------------------------
def js_divergence(real, fake):
    hist_real, _ = np.histogram(real, bins=50, density=True)
    hist_fake, _ = np.histogram(fake, bins=50, density=True)

    hist_real += 1e-8
    hist_fake += 1e-8

    m = 0.5 * (hist_real + hist_fake)

    return 0.5 * entropy(hist_real, m) + 0.5 * entropy(hist_fake, m)

# -----------------------------
# CORRELATION
# -----------------------------
def correlation(real, fake):
    return np.corrcoef(real[:10000], fake[:10000])[0, 1]

# -----------------------------
# COMPUTE METRICS
# -----------------------------
kl = kl_divergence(real_flat, fake_flat)
js = js_divergence(real_flat, fake_flat)
corr = correlation(real_flat, fake_flat)

print("\n===== CTGAN METRICS =====")
print("KL Divergence:", kl)
print("JS Divergence:", js)
print("Correlation:", corr)

# -----------------------------
# DIVERSITY
# -----------------------------
print("\n===== DIVERSITY =====")
print("Real Variance:", np.var(real_flat))
print("Generated Variance:", np.var(fake_flat))