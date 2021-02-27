import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from scipy import stats

#scipy version 1.6.1

num_models = 10

#array of mean error values for each model
pinn_means = np.array([0.0 for a in range(0,num_models)])
ltann_means = np.array([0.0 for a in range(0,num_models)])

#Loop through all 10 pairs of models for absolute value tests
for n in range (0,num_models):

    ##### Load predictions from pinn model N
    PINNData = np.load('PINNPredictions{}.npz'.format(n)) #load pinn prediction file
    labels = PINNData['predX'] #assign labels from the file, same values used for the label trained data
    pinn_predictions = PINNData['predY'] #assign predictions

    ##### Load predictions from label trained model N
    LTANNData = np.load('LTANNPredictions{}.npz'.format(n))
    lt_ann_predictions = LTANNData['predY'] #Assign label trained predictions

    # Mean of the absolute error of the predictions - labels
    mean_absolute_error_PINN = np.mean(abs(pinn_predictions - labels))
    mean_absolute_error_LTANN = np.mean(abs(lt_ann_predictions - labels))

    pinn_means[n] = mean_absolute_error_PINN #Add to vector containing mean prediction value from all predictions
    ltann_means[n] = mean_absolute_error_LTANN

## Scipy T test Returns T value, and two sided p value
print('Test with absolute value (prediction - label) values')
print(stats.ttest_rel(pinn_means,ltann_means)) #significant p value = pass
print('Pass')

# Let's test assumptions

#Normal distribution shapiro test, uses difference between 2 groups
mean_difference = pinn_means - ltann_means
print(stats.shapiro(mean_difference))
print('Fail') #significant p value = fail

#Anderson normality test
print(stats.anderson(mean_difference))
print('Fail') 

#Variance test
print(stats.levene(pinn_means,ltann_means))
print('Pass') #significant p value = pass

# mann whitney U test, alternative to t test
print(stats.mannwhitneyu(pinn_means,ltann_means))
print('Pass\n\n')

############################################
#Test again without absolute value, just copy and pasted the code from above but removed abs()

#Loop through all 10 pairs of models for absolute value tests
for n in range (0,num_models):

    ##### Load predictions from pinn model N
    PINNData = np.load('PINNPredictions{}.npz'.format(n)) #load pinn prediction file
    labels = PINNData['predX'] #assign labels from the file, same values used for the label trained data
    pinn_predictions = PINNData['predY'] #assign predictions

    ##### Load predictions from label trained model N
    LTANNData = np.load('LTANNPredictions{}.npz'.format(n))
    lt_ann_predictions = LTANNData['predY'] #Assign label trained predictions

    # Mean of the absolute error of the predictions - labels
    mean_absolute_error_PINN = np.mean((pinn_predictions - labels))
    mean_absolute_error_LTANN = np.mean((lt_ann_predictions - labels))

    pinn_means[n] = mean_absolute_error_PINN
    ltann_means[n] = mean_absolute_error_LTANN

## Returns T value, and two sided p value
print('Test with raw prediction - label values')
print(stats.ttest_rel(pinn_means,ltann_means))
print('Fail')

# Let's test assumptions

#Normal distribution shapiro test, uses difference between 2 groups
mean_difference = pinn_means - ltann_means
print(stats.shapiro(mean_difference))
print('Pass')

#Anderson normality test
print(stats.anderson(mean_difference))
print('Pass') 

#Variance test
print(stats.levene(pinn_means,ltann_means))
print('Pass')

print(stats.mannwhitneyu(pinn_means,ltann_means))
print('Fail\n\n')