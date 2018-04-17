# Load Libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
import argparse
import os

def main(path):
    print('Preparing data...')
    train = pd.read_csv(os.path.join(path,'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))

    df = pd.concat([train, test]).copy().reset_index(drop=True)
    df.smoking_status.fillna('not_answered',inplace=True)
    df.ever_married.replace({'Yes':1,'No':0},inplace=True)
    df.Residence_type.replace({'Urban':1,'Rural':0},inplace=True)
    df = pd.get_dummies(df)
    df['is_bmi_missed'] = np.where(pd.isnull(df.bmi),1,0)
    df.bmi.fillna(np.median(df[pd.notnull(df.bmi)].bmi),inplace=True)

    train = df.head(train.shape[0])
    test = df.tail(test.shape[0])

    cols = [x for x in train.columns if x not in ['stroke','bmi','gender_Other','work_type_Private']]

    X_train, y_train = train.loc[:,cols], train.stroke
    X_test = test.loc[:,cols]
    
    print('Catboost modelling...')
    def get_predictions(params):
        cv_data = cv(
        Pool(X_train, y_train),
        params,
        logging_level='Silent'
        )

        bst_iter = int(cv_data['test-AUC-mean'].idxmax() * 1.5)
        params['iterations'] = bst_iter

        model = CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
             logging_level='Silent'
        )

        return model.predict_proba(X_test)[:,-1]
    
    # Blend of 2 models
    params1 = {
    'iterations': 500,
    'depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'logging_level': 'Verbose',
    'l2_leaf_reg':15.0,
    'bagging_temperature':0.75
}

    params2 = {
    'iterations': 300,
    'depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'logging_level': 'Verbose'
    }

    pred = (get_predictions(params1)*get_predictions(params2))**(1/2)
    preds = pd.DataFrame({'id':test['id'],'stroke':pred})
    return preds
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input files', type=str, default=None)
    parser.add_argument('-o', '--output', help='Path to output csv', type=str, default='catboost_model.csv')
    args = parser.parse_args()
    
    preds = main(args.input)
    preds.to_csv(args.output,index=False)