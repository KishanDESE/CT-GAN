import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class ColumnInfo:
    def __init__(self, name, col_type, output_dim,
                 transformer, rev_transformer=None,
                 start=None, end=None):
        self.name = name
        self.col_type = col_type
        self.output_dim = output_dim
        self.transformer = transformer
        self.rev_transformer = rev_transformer
        self.start = start
        self.end = end


class DataTransformer:
    def __init__(self, continuous_cols, categorical_cols, n_clusters=5):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.n_clusters = n_clusters
        self.column_info = []
        self.output_dim = 0

    def fit(self, df: pd.DataFrame):
        self.column_info = []
        self.output_dim = 0

        for col in df.columns:
            if col in self.continuous_cols:
                gmm = GaussianMixture(
                    n_components=self.n_clusters,
                    covariance_type="full",
                    random_state=42
                )
                gmm.fit(df[col].values.reshape(-1, 1))

                info = ColumnInfo(
                    name=col,
                    col_type="continuous",
                    output_dim=1 + self.n_clusters,
                    transformer=gmm
                )

            elif col in self.categorical_cols:
                col_data = df[col].fillna("NaN").astype(str)
                categories = sorted(col_data.unique())

                mapping = {c: i for i, c in enumerate(categories)}
                rev_mapping = {i: c for c, i in mapping.items()}

                info = ColumnInfo(
                    name=col,
                    col_type="categorical",
                    output_dim=len(categories),
                    transformer=mapping,
                    rev_transformer=rev_mapping
                )

            else:
                raise ValueError(f"Column {col} not classified")

            info.start = self.output_dim
            info.end = self.output_dim + info.output_dim

            self.column_info.append(info)
            self.output_dim += info.output_dim

    def transform(self, df: pd.DataFrame):
        outputs = []

        for info in self.column_info:
            col = df[info.name]

            if info.col_type == "continuous":
                x = col.values.reshape(-1, 1)
                gmm = info.transformer

                probs = gmm.predict_proba(x)
                component = probs.argmax(axis=1)

                means = gmm.means_.reshape(-1)
                stds = np.sqrt(gmm.covariances_.reshape(-1))

                normalized = (x.flatten() - means[component]) / stds[component]
                normalized = np.tanh(normalized).reshape(-1, 1)

                onehot = np.zeros((len(x), self.n_clusters))
                onehot[np.arange(len(x)), component] = 1

                outputs.append(np.concatenate([normalized, onehot], axis=1))

            else:
                col = col.fillna("NaN").astype(str)
                mapping = info.transformer

                onehot = np.zeros((len(col), info.output_dim))
                idx = col.map(mapping).values
                assert not pd.isnull(idx).any(), f"Unknown category in {info.name}"

                onehot[np.arange(len(col)), idx] = 1
                outputs.append(onehot)

        return np.concatenate(outputs, axis=1)

    def inverse_transform(self, X: np.ndarray):
        data = {}

        for info in self.column_info:
            col_data = X[:, info.start:info.end]

            if info.col_type == "continuous":
                gmm = info.transformer

                tanh_val = np.clip(col_data[:, 0], -0.999, 0.999)
                mode = col_data[:, 1:].argmax(axis=1)

                normalized = np.arctanh(tanh_val)

                means = gmm.means_.reshape(-1)
                stds = np.sqrt(gmm.covariances_.reshape(-1))

                data[info.name] = normalized * stds[mode] + means[mode]

            else:
                indices = col_data.argmax(axis=1)
                data[info.name] = [info.rev_transformer[i] for i in indices]

        return pd.DataFrame(data)

# ===== Load Adult dataset =====


columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(
    "/home/yeshwant/CTGAN/adult/adult.data",
    header=None,
    names=columns,
    sep=",",
    skipinitialspace=True
)

df = df.replace("?", None)

continuous_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

categorical_cols = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country", "income"
]

def load_preprocessed_data():
    transformer = DataTransformer(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols
    )

    transformer.fit(df)
    X = transformer.transform(df)

    return X, transformer


if __name__ == "__main__":
    transformer = DataTransformer(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols
    )

    transformer.fit(df)
    X = transformer.transform(df)
    df_rt = transformer.inverse_transform(X)

    print("Original:")
    print(df.head())

    print("\nReconstructed:")
    print(df_rt.head())

    print("\nTransformed shape:", X.shape)

