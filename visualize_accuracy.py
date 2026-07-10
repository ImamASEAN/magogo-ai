import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load data dengan separator yang benar (;) dan decimal (,)
print("Memuat data...")
df = pd.read_csv('main_dataset.csv', sep=';', decimal=',')

# Tampilkan kolom yang tersedia untuk debugging
print(f"Kolom yang tersedia: {df.columns.tolist()}")

# Persiapan fitur berdasarkan nama kolom di dataset
# Mapping kolom dataset ke fitur yang kita butuhkan
feature_mapping = {
    'Feed_Amount': 'total_feed',
    'Temperature': 'avg_temperature',
    # Kita bisa tambahkan fitur lain jika relevan
}

target_col = 'Harvest_Mass'

# Filter kolom yang ada di dataset
available_features = [col for col in feature_mapping.keys() if col in df.columns]

if not available_features:
    print(f"Error: Tidak menemukan kolom fitur. Kolom yang tersedia: {df.columns.tolist()}")
    exit()

if target_col not in df.columns:
    print(f"Error: Kolom target '{target_col}' tidak ditemukan. Kolom yang tersedia: {df.columns.tolist()}")
    exit()

# Bersihkan data dari NaN pada kolom yang dipilih
df_clean = df[available_features + [target_col]].dropna()

if len(df_clean) == 0:
    print("Error: Tidak ada data valid setelah membersihkan NaN")
    exit()

print(f"Data valid: {len(df_clean)} baris")

X = df_clean[available_features]
y = df_clean[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
print("Melatih model Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data TEST (data yang belum pernah dilihat model)
y_pred_test = model.predict(X_test)

# Prediksi pada data FULL (untuk visualisasi keseluruhan tren)
y_pred_full = model.predict(X)

# Hitung Metrik
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print("\n--- METRIK AKURASI (Data Test) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print("----------------------------------")

# --- VISUALISASI 1: Scatter Plot Actual vs Predicted ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter points
ax.scatter(y_test, y_pred_test, color='#2E86AB', alpha=0.7, edgecolors='k', s=100, label='Data Test')
ax.scatter(y_train, model.predict(X_train), color='#A23B72', alpha=0.5, edgecolors='k', s=80, label='Data Latih', marker='s')

# Garis Perfect Prediction (y=x)
min_val = min(y.min(), y_pred_full.min())
max_val = max(y.max(), y_pred_full.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prediksi Sempurna (Akurasi 100%)')

ax.set_xlabel('Berat Panen Aktual (kg)', fontsize=12, fontweight='bold')
ax.set_ylabel('Berat Panen Prediksi AI (kg)', fontsize=12, fontweight='bold')
ax.set_title(f'AKURASI MODEL: Aktual vs Prediksi\n(MAE: {mae:.2f} kg | R²: {r2:.2%})', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# Tambahkan anotasi error
plt.text(0.05, 0.95, f'Semakin dekat titik ke garis merah,\nsemakin akurat prediksinya.', 
         transform=ax.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('8_actual_vs_predicted_scatter.png', dpi=300)
print("Grafik 1 tersimpan: 8_actual_vs_predicted_scatter.png")

# --- VISUALISASI 2: Line Chart Comparison (Tren) ---
# Urutkan berdasarkan indeks asli untuk melihat tren waktu/siklus
df_result = pd.DataFrame({
    'Aktual': y,
    'Prediksi': y_pred_full
})
df_result = df_result.sort_index()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df_result.index, df_result['Aktual'], marker='o', linestyle='-', color='#06A77D', label='Data Aktual', linewidth=2)
ax2.plot(df_result.index, df_result['Prediksi'], marker='x', linestyle='--', color='#F18F01', label='Prediksi AI', linewidth=2)

ax2.set_xlabel('Indeks Siklus / Data', fontsize=12, fontweight='bold')
ax2.set_ylabel('Berat Panen (kg)', fontsize=12, fontweight='bold')
ax2.set_title('PERBANDINGAN TREN: Data Aktual vs Prediksi AI', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('9_trend_comparison.png', dpi=300)
print("Grafik 2 tersimpan: 9_trend_comparison.png")

print("\nSelesai! Silakan cek file PNG yang dihasilkan untuk melihat detail akurasi.")
