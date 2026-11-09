# know data
import pandas as pd

df = pd.read_csv("data/kemiskinan.csv")
print(df.shape)
print(df.describe().round(2))
print(df[["p0", "tpt", "rls", "kota"]].corr()["p0"].round(3))
print(df.groupby("kota")[["p0", "tpt", "rls"]].mean().round(2))

# regression model
import numpy as np
from sklearn.linear_model import LinearRegression

x = df["tpt"].to_numpy()
y = df["p0"].to_numpy()
Sxy = np.sum((x - x.mean()) * (y - y.mean()))
Sxx = np.sum((x - x.mean()) ** 2)
b = Sxy / Sxx
a = y.mean() - b * x.mean()
pred = a + b * x
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"Rumus manual : p0 = {a:.4f} + ({b:.4f}) * tpt   R2 = {r2:.4f}")

b_np, a_np = np.polyfit(x, y, 1)
print(f"np.polyfit   : p0 = {a_np:.4f} + ({b_np:.4f}) * tpt")

lr = LinearRegression().fit(df[["tpt"]], y)
print(f"scikit-learn : p0 = {lr.intercept_:.4f} + ({lr.coef_[0]:.4f}) * tpt")

# visualisasi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, fitur, judul in [
    (axes[0], "tpt", "Pengangguran (TPT, %)"), 
    (axes[1], "rls", "Rata-rata lama sekolah (tahun)")]:
    for kota, warna, label in [(0, "tab:orange", "Kabupaten"), (1, "tab:blue", "Kota")]:
        bagian = df[df["kota"] == kota]
        ax.scatter(bagian[fitur], bagian["p0"], c=warna, label=label, alpha=0.7)
    kemiringan, potong = np.polyfit(df[fitur], df["p0"], 1)
    xs = np.linspace(df[fitur].min(), df[fitur].max(), 50)
    ax.plot(xs, potong + kemiringan * xs, "k--", label="garis regresi")
    ax.set_xlabel(judul)
    ax.set_ylabel("Penduduk miskin (%)")
    ax.legend()
fig.tight_layout()
fig.savefig("ml/kemiskinan/grafik.png", dpi=120)
print("Grafik disimpan ke ml/kemiskinan/grafik.png")

# cross-validation
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

cv = GroupKFold(n_splits=5)

def nilai(nama, model, fitur):
    pred = cross_val_predict(model, df[fitur], df["p0"], groups=df["wilayah"], cv=cv)
    mae = mean_absolute_error(df["p0"], pred)
    r2 = r2_score(df["p0"], pred)
    print(f"{nama:<28} MAE={mae:.3f} R2={r2:.3f}")
    return pred

nilai("Baseline (rata-rata)", DummyRegressor(), ["tpt"])

# add fitur
from sklearn.pipeline import make_pipeline           
from sklearn.preprocessing import StandardScaler     

for fitur in [["tpt"], ["rls"], ["tpt", "rls"], ["tpt", "rls", "kota"]]:
    nilai("Linear " + "+".join(fitur), make_pipeline(StandardScaler(), LinearRegression()), fitur)