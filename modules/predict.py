from datetime import datetime
import dill
import json
import pandas as pd
import os

path = os.environ.get('PROJECT_PATH', '.')
path_predictions = f'{path}/data/predictions'
path_models = f'{path}/data/models'  # путь до моделей
path_test = f'{path}/data/test'  # путь до тестовых файлов


def predict():
    # Функция выбора последней по дате модели
    def last_model(path_models):
        file_models = os.listdir(path_models)
        file_models_path = [os.path.join(path_models, f) for f in file_models]
        return max(file_models_path, key=os.path.getctime)

    with open(last_model(path_models), 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'price', 'pred'])

    for filename in os.listdir(path_test):
        with open(f'{path}/data/test/{filename}', 'r') as file_test:
            form = json.load(file_test)
            df = pd.DataFrame.from_dict([form])
            car_id = df.loc[0, 'id']
            price = df.loc[0, 'price']

            df_pred.loc[len(df_pred.index)] = [car_id, price, model.predict(df)[0]]


    df_pred.to_csv(f'{path_predictions}/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)



if __name__ == '__main__':
    predict()

