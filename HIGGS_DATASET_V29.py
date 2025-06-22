# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:00 2025

@author: TCSEKOC
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib

print(os.listdir('C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Final/'))

# Log dosyası ayarları
log_file = 'log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Loglama fonksiyonu
def log_action(message):
    logging.info(message)
    print(message)

# Rastgele örnekleme fonksiyonu
def sample_data_consistent_chunks(data_chunks, features_chunks, targets_chunks, sample_size=100000, random_state=None, chunk_size=5000):
    all_indices = []
    total_rows = sum(len(chunk) for chunk in data_chunks)
    if sample_size > total_rows:
        sample_size = total_rows
        log_action(f"Sample size {sample_size} olarak ayarlandı, toplam satır sayısından büyük olamaz.")

    current_index = 0
    for i, chunk in enumerate(data_chunks):
        chunk_size_local = len(chunk)
        chunk_indices = np.random.RandomState(random_state).choice(chunk_size_local, size=min(sample_size - len(all_indices), chunk_size_local), replace=False)
        global_indices = chunk_indices + current_index
        all_indices.extend(global_indices)
        current_index += chunk_size_local
        if len(all_indices) >= sample_size:
            break

    all_indices = np.array(all_indices)[:sample_size]
    data_parts = []
    features_parts = []
    targets_parts = []
    for i, chunk in enumerate(data_chunks):
        start_idx = sum(len(dc) for dc in data_chunks[:i])
        end_idx = start_idx + len(chunk)
        chunk_indices = all_indices[np.logical_and(all_indices >= start_idx, all_indices < end_idx)] - start_idx
        if len(chunk_indices) > 0:
            data_parts.append(chunk.iloc[chunk_indices])
            features_parts.append(features_chunks[i].iloc[chunk_indices])
            targets_parts.append(targets_chunks[i].iloc[chunk_indices])

    data = pd.concat(data_parts, axis=0, ignore_index=True)
    features = pd.concat(features_parts, axis=0, ignore_index=True)
    targets = pd.concat(targets_parts, axis=0, ignore_index=True)

    return data, features, targets

# Grafik çizme fonksiyonu
def plot_before_after(data_before, data_after, feature, step):
    if feature not in data_after.columns:
        log_action(f"Uyarı: {feature} sütunu veri setinde bulunmuyor, görselleştirme atlanıyor.")
        return
    
    data_before_vals = data_before[feature].values.reshape(-1, 1) if isinstance(data_before[feature], pd.Series) else data_before[feature].reshape(-1, 1)
    data_after_vals = data_after[feature].values.reshape(-1, 1) if isinstance(data_after[feature], pd.Series) else data_after[feature].reshape(-1, 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data_before_vals.ravel(), bins=30, kde=True)
    plt.title(f'Before {step} - {feature}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data_after_vals.ravel(), bins=30, kde=True)
    plt.title(f'After {step} - {feature}')
    
    plt.tight_layout()
    plt.savefig(f'before_after_{step}_{feature}.png')
    plt.close()

# Veri dağılımını görselleştirme fonksiyonu
def plot_feature_distributions(data, features, step='initial'):
    log_action(f"{step} veri dağılımı için histogramlar oluşturuluyor...")
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature].values.ravel(), bins=30, kde=True)
        plt.title(f'Distribution of {feature} ({step})')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(f'distribution_{step}_{feature}.png')
        plt.close()

# Model yükleme veya eğitme fonksiyonu
def load_or_train_model(model, params, X_train, y_train, X_val, y_val, model_filename):
    if os.path.exists(model_filename):
        log_action(f"{model_filename} bulundu, model yükleniyor...")
        return joblib.load(model_filename)
    else:
        log_action(f"{model_filename} bulunamadı, model eğitiliyor...")
        model_instance = model(**params)
        if model == XGBClassifier:
            model_instance.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model_instance.fit(X_train, y_train)
        joblib.dump(model_instance, model_filename)
        log_action(f"{model.__name__} modeli '{model_filename}' olarak kaydedildi.")
        return model_instance

# 1. Veri Ön İşleme
log_action("Bölüm 1: Veri Ön İşleme Başladı")

chunk_size = 25000
data_chunks = []
features_chunks = []
targets_chunks = []

log_action("Veri dosyaları 'higgs_dataset.csv', 'higgs_features.csv', 'higgs_targets.csv' parçalar halinde okunuyor...")
try:
    for chunk in pd.read_csv('higgs_dataset.csv', chunksize=chunk_size, engine='python', on_bad_lines='skip', dtype_backend='numpy_nullable'):
        data_chunks.append(chunk)
    for chunk in pd.read_csv('higgs_features.csv', chunksize=chunk_size, engine='python', on_bad_lines='skip', dtype_backend='numpy_nullable'):
        features_chunks.append(chunk)
    for chunk in pd.read_csv('higgs_targets.csv', chunksize=chunk_size, engine='python', on_bad_lines='skip', dtype_backend='numpy_nullable'):
        targets_chunks.append(chunk)
except Exception as e:
    log_action(f"Veri okuma hatası: {e}. Chunk size'ı düşürmeyi veya dosya yapısını kontrol etmeyi deneyin.")

if data_chunks and len(data_chunks[0]) != len(features_chunks[0]) or len(data_chunks[0]) != len(targets_chunks[0]):
    log_action("Uyarı: İlk parça satır sayıları eşleşmiyor!")
else:
    total_rows = sum(len(chunk) for chunk in data_chunks)
    log_action(f"Veri okundu. Toplam satır sayısı: {total_rows}, Sütun sayısı (dataset): {len(data_chunks[0].columns)}, (features): {len(features_chunks[0].columns)}, (targets): {len(targets_chunks[0].columns)}")

if data_chunks:
    data, features, targets = sample_data_consistent_chunks(data_chunks, features_chunks, targets_chunks, sample_size=100000, random_state=np.random.randint(0, 1000), chunk_size=5000)
    log_action(f"Rastgele {len(data)} satır ve eşleşen veriler seçildi. Yeni satır sayısı: {len(data)}")
else:
    log_action("Veri parçaları boş. İşlem durduruldu.")
    exit()

data_before = pd.concat([features, targets.iloc[:, 0].rename('target')], axis=1)
data = data_before.copy()
data.columns = ['target'] + [f'feature_{i}' for i in range(1, 29)]
log_action(f"Hedef değişkeni özelliklerle birleştirildi. Sütun sayısı: {len(data.columns)}")

data['target'] = (data['target'] > 0.5).astype(int)
log_action(f"Hedef değişken benzersiz değerler: {np.unique(data['target'])}")

# Veri dağılımını görselleştir
features = [f'feature_{i}' for i in range(1, 29)]
plot_feature_distributions(data, features, 'initial')

log_action("Aykırı değer analizi (IQR yöntemi, 3.0*IQR) başlatılıyor...")
def detect_outliers_chunk(df, iqr_multiplier=3.0):
    outliers = pd.DataFrame()
    outlier_counts = {}
    for column in features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_counts[column] = outliers[column].sum()
    log_action(f"Özellik bazında aykırı değer sayıları: {outlier_counts}")
    return outliers.any(axis=1)

outliers = detect_outliers_chunk(data, iqr_multiplier=3.0)
log_action(f"Aykırı değer tespit edildi: {outliers.sum()} adet aykırı değer bulundu.")

for feature in features[:15]:  # İlk 15 özellik için grafik
    plot_before_after(data_before, data, feature, 'outlier_removal')

for column in features:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    data.loc[data[column] < lower_bound, column] = lower_bound
    data.loc[data[column] > upper_bound, column] = upper_bound
log_action("Aykırı değerler sınır değerlerle değiştirildi.")

log_action("Özellik ölçekleme (MinMaxScaler) başlatılıyor...")
data_before_scaling = data.copy()
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

for feature in features[:15]:  # İlk 15 özellik için grafik
    plot_before_after(data_before_scaling, data, feature, 'scaling')

log_action("Özellikler [0, 1] aralığına ölçeklendirildi.")
log_action("Bölüm 1: Veri Ön İşleme Tamamlandı")

# 2. Özellik Seçimi ve Modelleme
log_action("Bölüm 2: Özellik Seçimi ve Modelleme Başladı")

feature_counts = [12, 13, 15]  # Sırayla 12, 13 ve 15 özellik için çalışacak
all_results = {}
best_params_dict = {}  # Her model ve özellik sayısı için en iyi parametreleri saklamak

# Outer ve Inner CV ayarları
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Hiperparametre ızgaraları
param_grids = {
    'KNN': [{'n_neighbors': n} for n in range(3, 12)],
    'SVM': [{'C': C, 'kernel': kernel, 'probability': True} for C in [0.1, 1, 10] for kernel in ['linear', 'rbf']],
    'MLP': [{'hidden_layer_sizes': size, 'activation': act} for size in [(50,), (100,)] for act in ['relu', 'tanh']],
    'XGBoost': [{'max_depth': d, 'learning_rate': lr, 'n_estimators': 100} for d in [3, 5] for lr in [0.01, 0.1]]
}

# Modeller
models = {
    'KNN': KNeighborsClassifier,
    'SVM': SVC,
    'MLP': MLPClassifier,
    'XGBoost': XGBClassifier
}

def nested_cv(X, y, model, param_grid, feature_count, model_name):
    outer_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'mcc': []}
    param_scores = {}  # Her hiperparametre kombinasyonu için ortalama skoru sakla
    model_filename = f'{model_name.lower()}_model_{feature_count}.pkl'

    # Model dosyasını kontrol et
    if os.path.exists(model_filename):
        log_action(f"{model_filename} bulundu, yalnızca performans değerlendiriliyor...")
        best_model = joblib.load(model_filename)
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            log_action(f"{model_name} için dış CV döngüsü {fold_idx + 1}/{outer_cv.n_splits} değerlendiriliyor...")
            X_test = X[test_idx]
            y_test = y[test_idx]
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            outer_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            outer_scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            outer_scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            outer_scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            outer_scores['roc_auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            outer_scores['mcc'].append(matthews_corrcoef(y_test, y_pred))
        return {key: np.mean(value) for key, value in outer_scores.items()}, None

    log_action(f"{model_filename} bulunamadı, tüm dış ve iç CV döngüleri çalıştırılıyor...")
    # Tüm dış CV döngülerini tamamla
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        log_action(f"{model_name} için dış CV döngüsü {fold_idx + 1}/{outer_cv.n_splits} başlatılıyor...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # İç CV ile her hiperparametre kombinasyonunu değerlendir
        for params in param_grid:
            param_key = str(params)  # Hiperparametreleri string olarak sakla
            if param_key not in param_scores:
                param_scores[param_key] = []
            log_action(f"{feature_count} özellik ile Inner CV için params: {params}")
            scores = []
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                X_inner_train, X_inner_val = X_train[inner_train], X_train[inner_val]
                y_inner_train, y_inner_val = y_train[inner_train], y_train[inner_val]
                X_inner_train_split, X_inner_val_split, y_inner_train_split, y_inner_val_split = train_test_split(
                    X_inner_train, y_inner_train, test_size=0.2, random_state=42
                )
                model_instance = model(**params)
                if model == XGBClassifier:
                    model_instance.fit(
                        X_inner_train_split, y_inner_train_split,
                        eval_set=[(X_inner_val_split, y_inner_val_split)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                else:
                    model_instance.fit(X_inner_train_split, y_inner_train_split)
                scores.append(model_instance.score(X_inner_val, y_inner_val))
            param_scores[param_key].append(np.mean(scores))

        # Dış CV için geçici model eğitimi ve performans
        # (Sadece performans değerlendirmesi için, kaydetme burada yapılmıyor)
        best_params_fold = max(param_scores, key=lambda k: np.mean(param_scores[k]))
        best_params_dict = eval(best_params_fold)  # String'den dict'e çevir
        model_instance = model(**best_params_dict)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        if model == XGBClassifier:
            model_instance.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model_instance.fit(X_train_split, y_train_split)
        y_pred = model_instance.predict(X_test)
        y_proba = model_instance.predict_proba(X_test)[:, 1]
        outer_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        outer_scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_scores['roc_auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        outer_scores['mcc'].append(matthews_corrcoef(y_test, y_pred))

    # Tüm döngülerden sonra en iyi hiperparametreleri seç ve modeli eğit
    best_params_key = max(param_scores, key=lambda k: np.mean(param_scores[k]))
    best_params = eval(best_params_key)
    log_action(f"{model_name} için tüm CV sonrası en iyi hiperparametreler: {best_params}")
    
    # En iyi hiperparametrelerle modeli bir kez eğit ve kaydet
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = load_or_train_model(model, best_params, X_train, y_train, X_val, y_val, model_filename)
    
    return {key: np.mean(value) for key, value in outer_scores.items()}, best_params

for k in feature_counts:
    log_action(f"{k} özellik ile modelleme başlatılıyor...")
    log_action(f"ANOVA F-score ile {k} özellik seçimi başlatılıyor...")
    X = data[[f'feature_{i}' for i in range(1, 29)]]  # Tüm 28 özellik
    y = data['target']
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = [features[i] for i in selector.get_support(indices=True)]
    log_action(f"Seçilen {k} özellik: {selected_features}")

    data_before_feature_selection = data.copy()
    data_selected = data[['target'] + selected_features]
    log_action(f"Seçilmiş {k} özellikli veri seti oluşturuldu.")

    # Sadece seçilen özellikler için görselleştirme (en fazla 15 özellik)
    for feature in selected_features[:15]:
        plot_before_after(data_before_feature_selection, data_selected, feature, f'feature_selection_{k}')

    # Model eğitimi ve değerlendirme
    X = data_selected.drop('target', axis=1).values
    y = data_selected['target'].values

    results = {}
    best_params_dict[k] = {}
    log_action(f"{k} özellik ile model eğitimi ve değerlendirmesi için nested CV başlatılıyor...")
    for name, model in models.items():
        log_action(f"{name} modeli için {k} özellik ile Nested CV çalışıyor...")
        results[name], best_params = nested_cv(X, y, model, param_grids[name], k, name)
        if best_params:
            best_params_dict[k][name] = best_params

    all_results[f'results_{k}'] = results

    # Sonuçları tablo olarak yazdır ve kaydet
    log_action(f"{k} özellik ile model performans sonuçları tablo olarak hazırlanacak...")
    results_df = pd.DataFrame(results).T
    print(f"\n{k} Özellik ile Model Performans Tablosu:")
    print(results_df)
    results_df.to_csv(f'model_performance_table_{k}.csv')
    log_action(f"{k} özellik ile model performans tablosu 'model_performance_table_{k}.csv' olarak kaydedildi.")

    # Grafikler için modelleri yükle
    log_action(f"{k} özellik ile ROC eğrileri çizimi başlatılıyor...")
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        model_filename = f'{name.lower()}_model_{k}.pkl'
        best_model = joblib.load(model_filename) if os.path.exists(model_filename) else None
        if best_model:
            log_action(f"{model_filename} bulundu, model yükleniyor...")
            fpr, tpr, _ = roc_curve(y, best_model.predict_proba(X)[:, 1])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y, best_model.predict_proba(X)[:, 1], multi_class="ovr"):.4f})')
        else:
            log_action(f"{model_filename} bulunamadı, grafik için model mevcut değil.")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for All Models ({k} Features)')
    plt.legend()
    plt.savefig(f'roc_curves_{k}_features.png')
    plt.show()

    log_action(f"{k} özellik ile Precision-Recall eğrileri çizimi başlatılıyor...")
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        model_filename = f'{name.lower()}_model_{k}.pkl'
        best_model = joblib.load(model_filename) if os.path.exists(model_filename) else None
        if best_model:
            log_action(f"{model_filename} bulundu, model yükleniyor...")
            precision, recall, _ = precision_recall_curve(y, best_model.predict_proba(X)[:, 1])
            plt.plot(recall, precision, label=f'{name} (AUC-PR = {roc_auc_score(y, best_model.predict_proba(X)[:, 1], multi_class="ovr"):.4f})')
        else:
            log_action(f"{model_filename} bulunamadı, grafik için model mevcut değil.")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for All Models ({k} Features)')
    plt.legend()
    plt.savefig(f'pr_curves_{k}_features.png')
    plt.show()

    log_action(f"{k} özellik ile Confusion Matrix çizimi başlatılıyor...")
    for name, model in models.items():
        model_filename = f'{name.lower()}_model_{k}.pkl'
        best_model = joblib.load(model_filename) if os.path.exists(model_filename) else None
        if best_model:
            log_action(f"{model_filename} bulundu, model yükleniyor...")
            y_pred = best_model.predict(X)
            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {name} ({k} Features)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name}_{k}.png')
            plt.show()
        else:
            log_action(f"{model_filename} bulunamadı, grafik için model mevcut değil.")

log_action("Bölüm 2: Özellik Seçimi ve Modelleme Tamamlandı")
log_action("Proje tamamlandı. Tüm loglar 'log.txt' dosyasına kaydedildi.")