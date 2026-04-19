import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc, f1_score

def gerar_metricas_finais():
    print("Carregando dados para avaliação final...")
    df = pd.read_csv('flattened_dataset.csv')
    X = df.filter(regex='^dim_').values
    y = df['syndrome_id'].values

    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    modelos = {
        'Euclidiana (K=10)': KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
        'Cosseno (K=15)': KNeighborsClassifier(n_neighbors=15, metric='cosine')
    }

    plt.figure(figsize=(10, 8))
    
    print("\n--- TABELA DE MÉTRICAS FINAIS ---")
    print(f"{'Modelo':<20} | {'F1-Score (Macro)':<18} | {'AUC (ROC)':<10}")
    print("-" * 55)

    for nome, modelo in modelos.items():
        y_prob = cross_val_predict(modelo, X, y, cv=kf, method='predict_proba')
        
        y_pred = classes[np.argmax(y_prob, axis=1)]

        f1 = f1_score(y, y_pred, average='macro')

        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        print(f"{nome:<20} | {f1:.4f}             | {roc_auc_micro:.4f}")

        plt.plot(fpr_micro, tpr_micro, lw=2, label=f'{nome} (AUC = {roc_auc_micro:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chute Aleatório (AUC = 0.500)')
    
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Comparação de Desempenho: Curva ROC (Cosseno vs Euclidiana)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    nome_imagem = 'roc_curve_comparison.png'
    plt.savefig(nome_imagem, dpi=300)
    print("\nGráfico gerado com sucesso: " + nome_imagem)

if __name__ == "__main__":
    gerar_metricas_finais()