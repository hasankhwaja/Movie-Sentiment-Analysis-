from imblearn.over_sampling import RandomOverSampler
import pandas as pd


train = pd.read_csv('train.tsv', sep = '\t')

# Separate features and target
X_train = train.drop("Sentiment", axis=1)
y_train = train["Sentiment"]

# Use RandomOverSampler to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Combine the resampled data
balanced_train_data = pd.concat([X_resampled, y_resampled], axis=1)

# Save the balanced train dataset
balanced_train_data.to_csv("balanced_train.tsv", sep="\t", index=False)
