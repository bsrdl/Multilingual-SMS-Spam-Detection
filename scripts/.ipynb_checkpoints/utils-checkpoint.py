import pandas as pd

def load_data(path):
    """
    Loads the dataset from the given path.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame
    """
    # Load dataset
    sms_spam_df = pd.read_csv(path)
    print(f"Dataset loaded from: {path}\n")

    # Display first 10 rows
    print("First 10 rows of the dataset:")
    display(sms_spam_df.head(10))

    # Dataset info
    print("\nDataset Info:")
    display(pd.DataFrame(sms_spam_df.dtypes, columns=["Data Type"]))

    # Missing values
    print("\nMissing values in each column: ")
    missing_values = sms_spam_df.isna().sum()
    display(missing_values)

    # Check duplicates
    duplicate_count = sms_spam_df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")

    # Spam distribution
    print("\nLabel distribution:")
    display(sms_spam_df["Label"].value_counts(normalize=True))

    return sms_spam_df
