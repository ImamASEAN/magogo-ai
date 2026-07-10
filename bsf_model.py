import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Nama fitur dasar (sesuai input user di app.py)
BASE_FEATURES = [
    'Feed_Amount', 'Bulking_Agent', 'Water_Added', 'pH_Substrate',
    'Ambient_Temp', 'Reactor_Temp', 'Day_Cycle', 'Frass_Collected'
]

# Nama fitur lengkap setelah engineering (harus sama persis dengan saat training)
ALL_FEATURES = BASE_FEATURES + [
    'Feed_Temp_Ratio', 'Feed_Water_Interact', 'Moisture_Index', 
    'Feed_Squared', 'Temp_Squared', 'Density_Proxy'
]

def create_features(df):
    """
    Fungsi untuk membuat fitur tambahan (Feature Engineering).
    Logika ini HARUS sama persis antara saat training dan prediction.
    """
    df = df.copy()
    
    # 1. Rasio Pakan terhadap Suhu (Efisiensi konversi termal)
    # Tambah epsilon kecil untuk menghindari pembagian dengan nol
    df['Feed_Temp_Ratio'] = df['Feed_Amount'] / (df['Reactor_Temp'] + 1e-6)
    
    # 2. Interaksi Pakan dan Air (Kelembaban substrat efektif)
    df['Feed_Water_Interact'] = df['Feed_Amount'] * df['Water_Added']
    
    # 3. Indeks Kelembaban (Water vs Total Massa Padat)
    total_solid = df['Feed_Amount'] + df['Bulking_Agent'] + 1e-6
    df['Moisture_Index'] = df['Water_Added'] / total_solid
    
    # 4. Fitur Kuadratik (Untuk menangkap pola non-linear)
    df['Feed_Squared'] = df['Feed_Amount'] ** 2
    df['Temp_Squared'] = df['Reactor_Temp'] ** 2
    
    # 5. Proksi Kepadatan Larva (Frass per hari)
    df['Density_Proxy'] = df['Frass_Collected'] / (df['Day_Cycle'] + 1e-6)
    
    return df

def train_model(df):
    """
    Melatih model Gradient Boosting dengan feature engineering.
    """
    # 1. Siapkan data dasar
    X_base = df[BASE_FEATURES].copy()
    y = df['Harvest_Mass'].copy()

    # 2. Buat fitur engineering
    X_eng = create_features(X_base)

    # 3. Imputasi (mengisi nilai kosong dengan rata-rata)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_eng), 
        columns=X_eng.columns, 
        index=X_eng.index
    )

    # 4. Latih Model Gradient Boosting
    # Parameter dioptimalkan untuk dataset kecil agar tidak overfitting parah
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_imputed, y)

    # Evaluasi sederhana pada data latih
    preds = model.predict(X_imputed)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    
    print(f"Model trained. R²: {r2:.4f}, MAE: {mae:.2f} kg")

    # Simpan bundle (model, imputer, daftar fitur)
    bundle = {
        'model': model,
        'imputer': imputer,
        'features': ALL_FEATURES,
        'metrics': {'r2': r2, 'mae': mae}
    }
    
    return bundle

def predict_single(bundle, input_dict):
    """
    Melakukan prediksi untuk satu data input baru.
    """
    model = bundle['model']
    imputer = bundle['imputer']
    expected_features = bundle['features']
    
    # 1. Ubah input dictionary menjadi DataFrame
    # Pastikan urutan kolom sesuai BASE_FEATURES dulu
    row = pd.DataFrame([input_dict], columns=BASE_FEATURES)
    
    # 2. TERAPKAN FEATURE ENGINEERING (PENTING!)
    # Ini menambahkan kolom-kolom baru yang dibutuhkan model
    row_eng = create_features(row)
    
    # 3. Pastikan urutan kolom sesuai dengan yang diharapkan model (ALL_FEATURES)
    # Jika ada kolom yang hilang, diisi NaN dulu
    for col in expected_features:
        if col not in row_eng.columns:
            row_eng[col] = np.nan
            
    # Re-index agar urutannya persis sama dengan saat training
    row_final = row_eng[expected_features]
    
    # 4. Imputasi (mengisi nilai NaN jika ada)
    row_imputed = pd.DataFrame(
        imputer.transform(row_final),
        columns=expected_features
    )
    
    # 5. Prediksi
    prediction = model.predict(row_imputed)[0]
    
    # Pastikan hasil tidak negatif (secara biologis tidak mungkin)
    return max(0.0, prediction)

def save_model(bundle, path='bsf_model.joblib'):
    joblib.dump(bundle, path)
    print(f"Model saved to {path}")

def load_model(path='bsf_model.joblib'):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Script untuk menjalankan training jika file ini dijalankan langsung
if __name__ == "__main__":
    # Cek apakah ada dataset
    if os.path.exists('main_dataset.csv'):
        df = pd.read_csv('main_dataset.csv')
        
        # Bersihkan data sederhana
        required_cols = BASE_FEATURES + ['Harvest_Mass']
        df = df.dropna(subset=required_cols)
        
        if len(df) > 10:
            print("Memulai pelatihan model Gradient Boosting...")
            bundle = train_model(df)
            save_model(bundle)
            print("Pelatihan selesai!")
        else:
            print("Data terlalu sedikit untuk melatih model.")
    else:
        print("File main_dataset.csv tidak ditemukan.")
