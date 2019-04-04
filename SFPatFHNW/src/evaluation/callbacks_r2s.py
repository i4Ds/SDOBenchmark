from sklearn.metrics import r2_score
from functools import reduce

def r2_on_epoch_end(original_callback, epoch, y_true, y_pred):
    """
    The R2-Score, often also called the Coefficient of Determination, measures the proportion of the variance in
    the dependent variable that is predictable from the independent variables. It is calculated as follows:

    $$\text{R}^2=1 − \frac{\sum^n_{i=1} {(y_i−\hat{y}_i)^2}}{ \sum^n_{i=1} {(y_i−\bar{y})^2}}$$

    Another way to interpret the R2-Score is the following: It is the percentage of the distance between a
    model always predicting the median value (the baseline model) and a perfect model, that the evaluated model
    has covered. Therefore, a perfect model would achieve a R2-Score of 1.  A model with a R2-Score of 0 has
    comparable performance than the baseline. Finally, a score below 0 indicates that the model evaluated is worse
    than baseline.
    """
    
    original_callback.add_simple_value('R2', r2_score(y_true, y_pred), epoch)

    return

def adjustedR2FuncFactory(featureDims):
    featuresCount = reduce(lambda x, y: x*y, featureDims)
    
    def ar2_on_epoch_end(original_callback, epoch, y_true, y_pred):
        
        n = len(y_true)
        enum = (1 - r2_score(y_true,y_pred))  * (n - 1.0)
        denom = (n - featuresCount - 1)
        
        original_callback.add_simple_value('Adjusted R2', 1.0 - enum/denom, epoch)
        
        return

    return ar2_on_epoch_end
