from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def mae_on_epoch_end(original_callback, epoch, y_true, y_pred):
    
    original_callback.add_simple_value('MAE', mean_absolute_error(y_true, y_pred), epoch)

    return

def mse_on_epoch_end(original_callback, epoch, y_true, y_pred):
    """
    The MSE (Mean Squared Error) is calculated very similar to the MAE:

	$$\text{MSE}=\frac{\sum^n_{i=1}{(\hat{y}_i−y_i)^2}}{n}$$

    Where once again, $$n$$ is the number of samples in the validation data set and $$\hat{y}_i$$ the prediction
    and $$y_i$$ the true label for the i-th sample.
    In comparison to the MAE, the MSE takes into account both the variance of the predictions as well as theirs biases.
    Another important difference is, that it uses the square unit of measurement of the predictions. The main drawback
    of the MSE in this comparison, is that outliers are weight very heavily.
    Using the RMSE (Root Mean Squared Error) returns the scale and unit back to that of the predictions itself,
    however, the overly heavy weighing of outliers is still a property of this metric.

	$$\text{RMSE}=\sqrt{\frac{\sum^n_{i=1}{(\hat{y}_i−y_i)^2}}{n}}$$
    """

    original_callback.add_simple_value('MSE', mean_squared_error(y_true, y_pred), epoch)
    
    return

def rmse_on_epoch_end(original_callback, epoch, y_true, y_pred):
    
    original_callback.add_simple_value('RMSE', sqrt(abs(mean_squared_error(y_true, y_pred))), epoch)
    
    return

