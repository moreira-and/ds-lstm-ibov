import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

# Custom metrics for regression
# SMAPE - Symmetric Mean Absolute Percentage Error
@register_keras_serializable()
def smape(y_true, y_pred):
    numerator = K.abs(y_true - y_pred)
    denominator = K.maximum((K.abs(y_true) + K.abs(y_pred)) / 2.0, K.epsilon())
    return 100.0 * K.mean(numerator / denominator)
smape.__name__ = "smape"

# RMSE - Root Mean Squared Error
@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
rmse.__name__ = "rmse"

# R² Score - Coefficient of Determination
@register_keras_serializable()
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - ss_res / K.maximum(ss_tot, K.epsilon())
r2_score.__name__ = "r2_score"