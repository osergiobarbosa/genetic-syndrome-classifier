import pickle
import pandas as pd
import numpy as np

file_path = 'mini_gm_public_v0.1.p'
def process_data():
    print("1. Carregando o arquivo pickle...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("2. Achatando a estrutura hierárquica (Flattening)...")
    rows = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                
                row = {
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id
                }
                
                for i, value in enumerate(embedding):
                    row[f'dim_{i}'] = value
                    
                rows.append(row)

    df = pd.DataFrame(rows)
    
    print("3. Verificando a integridade dos dados...")

    missing_data = df.isnull().sum().sum()
    print(f"   Valores ausentes encontrados: {missing_data}")
    
    if missing_data > 0:
        df = df.dropna()
        print("   Valores ausentes removidos.")

    print(f"\nResumo: O dataset final possui {df.shape[0]} imagens e {df.shape[1]} colunas.")
    
    output_file = 'flattened_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"Dados salvos com sucesso em: '{output_file}'")

if __name__ == "__main__":
    process_data()