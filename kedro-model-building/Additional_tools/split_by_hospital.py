def split_data_by_hospital(df):
  """Splits the data in df into train and test data based on hospital_id.

  Args:
    df: The pandas DataFrame to split.

  Returns:
    A tuple of two DataFrames, the first containing the train data and the
    second containing the test data.
  """

  train_df = pd.DataFrame()
  test_df = pd.DataFrame()

  for hospital_id, group in df.groupby("hospital_id"):
    train_size = int(0.6 * len(group))
    train_df = pd.concat([train_df, group.iloc[:train_size]], ignore_index=True)
    test_df = pd.concat([test_df, group.iloc[train_size:]], ignore_index=True)

  return train_df, test_df
