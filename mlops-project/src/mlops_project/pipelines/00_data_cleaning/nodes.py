import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove leading spaces from column names
    df.columns = df.columns.str.strip()

    # Clean categorical columns by stripping spaces from string values
    categorical_columns = ["education", "self_employed", "loan_status"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Convert 'no_of_dependents' and 'loan_term' to string (categorical)
    if "no_of_dependents" in df.columns:
        df["no_of_dependents"] = df["no_of_dependents"].astype(str).str.strip()

    if "loan_term" in df.columns:
        df["loan_term"] = df["loan_term"].astype(str).str.strip()

    # Map and drop 'education'
    if "education" in df.columns:
        df["Graduate"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
        df.drop(columns=["education"], inplace=True)

    # Map values for 'self_employed' and keep the column
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    # Map and drop 'loan_status'
    if "loan_status" in df.columns:
        df["loan_approved"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
        df.drop(columns=["loan_status"], inplace=True)

    return df