import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def calcular_top_k_accuracy_manual(y_verdadeiro, y_probabilidades, classes_do_modelo, k=3):
    """
    Implementação manual do Top-k Accuracy.
    Verifica se a classe real está entre as 'K' maiores probabilidades previstas.
    """
    acertos = 0
    total = len(y_verdadeiro)
    
    for i in range(total):
        probs = y_probabilidades[i]
        top_k_indices = np.argsort(probs)[-k:]
        top_k_classes = [classes_do_modelo[idx] for idx in top_k_indices]
        if y_verdadeiro[i] in top_k_classes:
            acertos += 1
            
    return acertos / total

def executar_modelo():
    print("1. Carregando os dados...")
    df = pd.read_csv('flattened_dataset.csv')
    X = df.filter(regex='^dim_').values
    y = df['syndrome_id'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    distancias = ['euclidean', 'cosine']
    valores_k = range(1, 16)
    
    print("\n2. Iniciando treinamento e validação cruzada...")
    print("Aguarde, testando dezenas de combinações (K e Distâncias)...\n")
    melhor_acuracia = 0
    melhor_k = 0
    melhor_distancia = ""

    for dist in distancias:
        print(f"--- Avaliando métrica de distância: {dist.upper()} ---")
    
        for k in valores_k:
            acuracias_fold = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                knn = KNeighborsClassifier(n_neighbors=k, metric=dist)
                knn.fit(X_train, y_train)
                
                y_pred = knn.predict(X_test)
                
                acc = np.mean(y_pred == y_test)
                acuracias_fold.append(acc)
            media_acc = np.mean(acuracias_fold)
            
            if media_acc > melhor_acuracia:
                melhor_acuracia = media_acc
                melhor_k = k
                melhor_distancia = dist
                
            print(f"K = {k:2d} | Acurácia Média: {media_acc:.4f}")
        print("-" * 40)

    print("\n" + "="*50)
    print("RESULTADO DA BUSCA PELO MELHOR MODELO:")
    print(f"Melhor Distância: {melhor_distancia.upper()}")
    print(f"Melhor K        : {melhor_k}")
    print(f"Maior Acurácia  : {melhor_acuracia * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    executar_modelo()