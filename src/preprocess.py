import pandas as pd 
import deepchem as dc 
from sklearn.model_selection import train_test_split



def balance_dataset(df):

    positive=df[df['HIV_active']==1]
    negative=df[df['HIV_active']==0]

    ###Undersample the negatve class as that causes the class imbalance in the
    #dataset 

    negative_sampled=negative.sample(n=len(positive), random_state=42)

    balance_df=pd.concat([positive,negative_sampled])
    return balance_df.sample(frac=1, random_state=42).reset_index(drop=True)


def feature_data(csv_path:str):

    #load the data 

    df= pd.read_csv(csv_path)
    df= df.dropna()

    #Converting SMILES to molecular data using deepChem featurizer

    df= balance_dataset(df)

    smiles=df['smiles']
    labels=df['HIV_active']

    featurizer = dc.feat.ConvMolFeaturizer()
    features= featurizer.featurize(smiles)

    X_train, X_test, y_train, y_test =train_test_split(features,labels, test_size=0.2, random_state=42)

    #creating DeepChem dataset

    train_dataset= dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test,y_test)

    return train_dataset, test_dataset 

if __name__ == "__main__":
    train_data, test_data = feature_data("C:\\Internship\\DRUG_DISCOVERY_HIV_INHIBITOR\\dataset\\HIV.csv")
    print("Train samples:", len(train_data))
    print("Test samples:", len(test_data))