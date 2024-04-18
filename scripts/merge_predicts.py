import sys
import pandas as pd
from tqdm import tqdm


if __name__== '__main__':
    result = sys.argv[1]

    table = sum([pd.get_dummies(pd.read_csv(name), columns=['target_feature'], dtype=int) for name in sys.argv[2:]])

    rows = []
    features = [f'target_features_{i}' for i in range(3)]
    for _, row in tqdm(table.iterrows(), desc="Merge predictions"):
        result_id = row.iloc[0]
        result_target = row[1:].idxmax()
        rows.append([result_id, result_target])
    
    result_df = pd.DataFrame(rows, columns=['id', 'target_feature'])
    

# Подсчет моды для каждого уникального значения Id
#mode_predictions = merged_df.groupby('Id')['Target_Feature'].apply(lambda x: x.mode()[0])

# Запись результата в выходной файл
#output_file = 'most_common_predictions.csv'
#mode_predictions.to_csv(output_file, header=['Target_Feature'])

