import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Clean and preprocess the data."""
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Normalize numerical features
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_imputed)

    return data_normalized

def split_data(data, target_column):
    """Split the data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Example usage
    data = load_data('data/raw/healthcare_data.csv')
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data, 'target')
    
    # Save processed data
    pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
