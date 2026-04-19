import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def executar_analise_visual():
    print("1. A carregar os dados formatados...")
    df = pd.read_csv('flattened_dataset.csv')

    print("\n--- Estatísticas do Dataset ---")
    total_sindromes = df['syndrome_id'].nunique()
    print(f"Total de Síndromes distintas: {total_sindromes}")
    
    print("\nContagem de imagens por Síndrome (Verificando o balanceamento):")
    contagem = df['syndrome_id'].value_counts()
    print(contagem)
    print("-" * 30)

    y = df['syndrome_id']
    X = df.filter(regex='^dim_')

    print("\n2. A aplicar o algoritmo t-SNE (isto pode demorar alguns segundos)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]

    print("3. A gerar o gráfico 2D...")
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        x='tsne_x', y='tsne_y',
        hue='syndrome_id',
        palette='tab10',
        data=df,
        alpha=0.7
    )
    
    plt.title('Visualização t-SNE: Agrupamento de Síndromes Genéticas')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Síndromes")
    plt.tight_layout()

    nome_imagem = 'tsne_grafico.png'
    plt.savefig(nome_imagem, dpi=300)
    print(f"\nSucesso! O gráfico foi guardado como '{nome_imagem}'.")

if __name__ == "__main__":
    executar_analise_visual()