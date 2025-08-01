import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame({
    'num': [1, 2, 3],
    'cat': ['a', 'b', 'a']
})

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['num']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['cat'])
    ],
    remainder='passthrough'
)

X_trans = preprocessor.fit_transform(df)
X_inv = preprocessor.transformers.inverse_transform(X_trans)

print(pd.DataFrame(X_inv, columns=['num', 'cat']))
