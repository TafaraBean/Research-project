

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
import statsmodels.stats.outliers_influence as smoi
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import ttest_ind, levene
from scipy.stats import chi2_contingency


###Phase 1
###Tafara
###Extracting the data
data = pd.read_csv("C:/pythonproj/sport_perform.csv")
data_swim = data[data["sport"]=="Swimming"]


### getting the binary gender
gender_binary=data_swim['gender'].map({'Male':1,'Female':0})
data_swim = data_swim.drop('gender',axis=1)
data_swim.insert(4,'gender',gender_binary)
   
### Extracting the design matrix
xvar = data_swim.iloc[:,4:]


### extracting the performance vector
perform = data_swim['perform']


###inserting the performance variable in the design matrix
xvar.insert(0,'perform',perform)


###computing the correlation matrix
correl = xvar.corr()
'''
There seems to be no significant relationships for the gender
'''
###get the performance correlations
perform_corr = correl.iloc[0,:] 




###get the correlations which are 0.50<
perform_corr = perform_corr[perform_corr>0.30]




### getting the upper triangle as a boolean matrix
upper = np.triu(np.ones(correl.shape), k=1).astype(bool)


###getting the actual values in the upper triangle
filtered_cor = correl.where(upper)


###getting the entries which are perfectly correlated
high_cor = filtered_cor[filtered_cor==1]


'''
From this we notice that variable x13 and x22 measure the exact same information.
Fitness and Physical fitness are exactly the same. hence we can remove physical fitness
'''
### dropping the physical fitness column and recomputing correlations
data_swim=data_swim.drop('x22',axis=1)
xvar = data_swim.iloc[:,4:]
perform = data_swim['perform']
xvar.insert(0,'perform',perform)
correl = xvar.corr()
perform_corr = correl.iloc[0,:] 
perform_corr = perform_corr[perform_corr>0.30]




### remap the gender variable
gender_binary=data_swim['gender'].map({1:'Male',0:'Female'})
data_swim = data_swim.drop('gender',axis=1)
data_swim.insert(4,'gender',gender_binary)


###exploratory plots(This is where the exploration is done!)
#==========================================================================
###Scatter plot investigating performance and endurance relationship
###functions to create categorical variables




#Tafara creates categorical variables

def determine_size(x):
    if x < 35:
        return 1
    elif 35 <= x < 50:
        return 2
    else:
        return 3
    
data_swim['fitness_level'] = data_swim['x13'].apply(determine_size)
data_swim['oxygen_uptake'] = data_swim['x18'].apply(determine_size)
data_swim['LTL'] = data_swim['x16'].apply(determine_size)
data_swim['endurance_size'] = data_swim['x11'].apply(determine_size)
data_swim['muscle_category']= data_swim['x20'].apply(determine_size)
data_swim['skill_level']= data_swim['x28'].apply(determine_size)
data_swim['mental_level']= data_swim['x19'].apply(determine_size)


#Neo plot 
# 1. Boxplot with titles, labels, and colors
sns.boxplot(data=data_swim, x='gender', y='perform', hue='muscle_category', palette='Set2')
plt.title('Performance by Gender and Muscle Category')
plt.xlabel('Gender')
plt.ylabel('Performance')
plt.legend(title='Muscle Category', loc='upper right')
plt.show()


# 2. Countplot with titles, labels, and colors
sns.countplot(data=data_swim, x='gender', hue='muscle_category', palette='Set1')
plt.title('Frequency of Muscle Category by Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.legend(title='Muscle Category', loc='upper right')
plt.show()




# 3. Scatterplot (with hue) with titles, labels, and colors
sns.scatterplot(data=data_swim, x='x20', y='perform', hue='mental_level', palette='coolwarm')
plt.title('Muscle Strength vs. Performance by Mental Level')
plt.xlabel('Muscle Strength (x20)')
plt.ylabel('Performance')
plt.legend(title='Mental Level', loc='upper left')
plt.show()


# 4. Linear regression plot with titles and labels
sns.lmplot(data=data_swim, x='x20', y='perform', aspect=1.5, scatter_kws={'alpha':0.5})
plt.title('Linear Regression of Muscle Strength vs. Performance')
plt.xlabel('Muscle Strength (x20)')
plt.ylabel('Performance')
plt.show()








###Tafara Plots
###Scatter plot investigating performance and endurance
plt.scatter(data_swim['perform'],data_swim['x11'])
plt.xlabel('endurance')
plt.ylabel('perform')
plt.show()


sns.lmplot(data_swim,x='perform',y='x11')
plt.ylabel('perform')
plt.xlabel('endurance')
plt.show()




###Scatter plot investigating performance and maximal oxygen uptake
plt.scatter(data_swim['perform'],data_swim['x18'])
plt.xlabel('maximal oxygen uptake')
plt.ylabel('perform')
plt.show()


sns.lmplot(data_swim,x='perform',y='x18')
plt.ylabel('perform')
plt.xlabel('maximal oxygen uptake')
plt.show()




#the difference in genders
sns.countplot(data_swim,x='gender',hue='gender')
plt.show()




#boxplot for gender and fitness
sns.boxplot(data_swim,x='gender',y='x13')
plt.show()


#whats the combined relationships?
#bubble plot for endurance vs performance, 3D=fitness, 4D=oxygen uptake
sns.scatterplot(data_swim ,x=data_swim['x11'].astype(float),y=perform.astype(float),hue='fitness_level',size=data_swim['oxygen_uptake'])
plt.xlabel('endurance')
plt.show()


#bubble plot for endurance vs performance, 3D=fitness, 4D=lactate_threshold
sns.scatterplot(data_swim ,x=data_swim['x11'].astype(float),y=perform.astype(float),hue='fitness_level',size=data_swim['LTL'])
plt.xlabel('endurance')
plt.show()




# Unathi plots
# Scatter plot for performance vs fitness
plt.scatter(data_swim['perform'], data_swim['x13'], color='blue')
# Add title and axis labels
plt.title('Relationship between Performance and Fitness')
plt.xlabel('Fitness (x13)')
plt.ylabel('Performance')
plt.show()


# Linear regression plot to observe trend between performance and fitness
sns.lmplot(data=data_swim, x='perform', y='x13', scatter_kws={'color':'blue'}, line_kws={'color':'red'})
# Add title and axis labels
plt.title('Linear Regression: Performance vs Fitness')
plt.xlabel('Performance')
plt.ylabel('Fitness')
plt.show()


# Scatter plot for performance vs lactate threshold
plt.scatter(data_swim['perform'], data_swim['x16'], color='purple')
# Add title and axis labels
plt.title('Relationship between Performance and Lactate Threshold')
plt.xlabel('Lactate Threshold (x16)')
plt.ylabel('Performance')
plt.show()


# Linear regression plot for performance vs lactate threshold
sns.lmplot(data=data_swim, x='perform', y='x16', scatter_kws={'color':'purple'}, line_kws={'color':'black'})
# Add title and axis labels
plt.title('Linear Regression: Performance vs Lactate Threshold')
plt.xlabel('Performance')
plt.ylabel('Lactate Threshold')
plt.show()


# Linear regression plot for maximal oxygen uptake vs fitness with hue for endurance size
sns.lmplot(data=data_swim, x='x18', y='x16', hue='endurance_size', palette='Set2', aspect=1.5, scatter_kws={'alpha':0.5})
# Add title and axis labels
plt.title('Maximal Oxygen Uptake vs Fitness by Endurance Size')
plt.xlabel('Maximal Oxygen Uptake (x18)')
plt.ylabel('Fitness (x16)')
# Add legend for endurance size categories
plt.legend(title='Endurance Size')
plt.show()


# Boxplot showing variation in performance by fitness level
sns.boxplot(data=data_swim, x='fitness_level', y='perform', hue='fitness_level', palette='coolwarm')
# Add title and axis labels
plt.title('Performance Variation by Fitness Level')
plt.xlabel('Fitness Level')
plt.ylabel('Performance')
# Add legend for fitness level
plt.legend(title='Fitness Level')
plt.show()


# Boxplot showing lactate threshold by endurance size and fitness level
sns.boxplot(data=data_swim, x='endurance_size', y='x16', hue='fitness_level', palette='Set1')
# Add title and axis labels
plt.title('Lactate Threshold by Endurance Size and Fitness Level')
plt.xlabel('Endurance Size')
plt.ylabel('Lactate Threshold (x16)')
# Add legend for fitness level
plt.legend(title='Fitness Level')
plt.show()


# Boxplot showing lactate threshold variation by fitness level
sns.boxplot(data=data_swim, x='fitness_level', y='perform', hue='fitness_level', palette='Set3')
# Add title and axis labels
plt.title('Lactate Threshold Variation by Fitness Level')
plt.xlabel('Fitness Level')
plt.ylabel('Lactate Threshold')
# Add legend for fitness level
plt.legend(title='Fitness Level')
plt.show()


# Boxplot for oxygen uptake vs lactate threshold level by fitness level
sns.boxplot(data=data_swim, x='LTL', y='x18', hue='fitness_level', palette='coolwarm')
# Add title and axis labels
plt.title('Oxygen Uptake by Lactate Threshold Level and Fitness Level')
plt.xlabel('Lactate Threshold Level (LTL)')
plt.ylabel('Maximal Oxygen Uptake (x18)')
# Add legend for fitness level
plt.legend(title='Fitness Level')
plt.show()






'''
Note*
people with a higher oxygen uptake have a higher lactate threshold level
'''
print('reached end of phase 1')
###Phase 2

###Tafara start

def levels(x):
    if x < 50:
        return 'low'
    elif 50 <= x < 75:
        return 'Medium'
    else:
        return 'high'


data_swim['perform_levels'] = data_swim['perform'].apply(levels)
data_swim['psycho_level'] = data_swim['x23'].apply(levels)


tab = pd.crosstab(data_swim['perform_levels'],data_swim['psycho_level'])


chi2_contingency(tab)      



'''
H0: data is independent vs HA: data is dependent


p-value: 0.5022809583952252


Decision: DO NOT reject the null hypothesis


Conclusion: performance levels are NOT dependent on the psychological levels 
'''  

###graphs

#Neo graphs

sns.lmplot(data_swim,x='perform',y='x23')
plt.show()


#there is a weak relationship between these 2 variables


sns.boxplot(data_swim,x='psycho_level',y='perform')
plt.show()
#they overlap. no significant difference




sns.boxplot(data_swim,x='perform_levels',y='x23')
plt.show()
#they overlap


tab_correl=tab.corr()


#show a high correlation amongst levels


sns.heatmap(tab_correl,annot=False,cmap='coolwarm')
plt.ylabel('perform_level')
plt.show()
#no significant differences


sns.barplot(data_swim,x='psycho_level',y=perform)
plt.show()
#no difference


sns.barplot(data_swim,x='perform_levels',y='x23')
plt.show()
#differences are not that significant


###Gender tests 
#Tafara splits populations
perform_M = data_swim.perform[data_swim.gender=='Male']
perform_F = data_swim.perform[data_swim.gender=='Female']


perform_M_new =(perform_M-np.mean(perform_M))/np.std(perform_M,ddof=1)
perform_F_new =(perform_F-np.mean(perform_F))/np.std(perform_F,ddof=1)


pool = np.concatenate([perform_F_new,perform_M_new])

#Tafara tests
ans=shapiro(pool)


'''
H0: pooled data is normally distributed vs. HA: pooled data is not normally distributed


p-value: 0.12197036752169904


Decision: DO NOT reject the null hypothesis


Conclusion: the data is normally distributed
'''


ans2 = levene(perform_M,perform_F)


'''
H0: there is homoskedasticity vs HA: there is heteroskedacity


p-value: 0.8429616887639799


Decision: DO NOT reject the null hypothesis


Conclusion: the data is homoskedastic
'''

#Neo tests
ans3 = ttest_ind(perform_M, perform_F,equal_var=True,alternative='two-sided')


'''
H0: the mean of male performance is equal to the mean female performance
vs.
HA: the mean of male performance is unequal to the mean female performance


p-value:0.4845800741844959


Decision: DO NOT reject the null hypothesis


Conclusion: the mean male performance is not significantly different from
            the female mean perfromance
'''


#Unathi graphs
sns.boxplot(data_swim,x='gender',y=perform,hue='gender')
plt.show()


sns.barplot(data_swim,x='gender',y=perform,hue='gender')
plt.show()


print('reached end of phase 2.')


####Phase 3

###Tafara
xvar = data_swim.iloc[:,4:32]
xvar.insert(0,'perform',perform)
formula_str = "perform ~ " + " + ".join(xvar.columns.difference(["perform"]))
mod = smf.ols(formula= formula_str, data=xvar).fit()
mod.summary()


###
###choosing the model
mod2 = smf.ols(formula= 'perform ~ gender + x07 + x09 + x11 + x16 + x20 + x21 + x28', data=xvar).fit()
mod2.summary()


xvar = data_swim.iloc[:,4:32]
xvar.insert(0,'perform',perform)
xvar = xvar[xvar['gender']=='Male']
mod3 = smf.ols(formula= 'perform ~ x07 + x09 + x11 + x16 + x20 + x21 + x28', data=xvar).fit()
mod3.summary()


xvar = data_swim.iloc[:,4:32]
xvar.insert(0,'perform',perform)
xvar = xvar[xvar['gender']=='Female']
mod4 = smf.ols(formula= 'perform ~ x07 + x09 + x11 + x16 + x20 + x21 + x28', data=xvar).fit()
mod4.summary()


# now why would these be the pivotal variables?
#-------------------------------------------------------------------------------------------------------------
## ****** Unathi *******
xvar = data_swim.iloc[:,4:32]
xvar.insert(0,'perform',perform)
formula_str = "perform ~ " + " + ".join(xvar.columns.difference(["perform"]))
mod = smf.ols(formula= formula_str, data=xvar).fit()
mod.summary()


###
###choosing the model
mod2 = smf.ols(formula= 'perform ~ gender + x07 + x09 + x11 + x16 + x20 + x21 + x28', data=xvar).fit()




residuals = mod2.resid
fitted_values = mod2.fittedvalues


# Multicollinearity Test
noYdat = mod2.model.exog
vif=[smoi.variance_inflation_factor(noYdat, i) for i in [1,2,3,4,5,6,7,8]]
print(vif)


# NORMALITY TESTS 
# (graphical)
# Q-Q Plot of the residuals
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid()
plt.show()


# Histogram of residuals with normal and kernel density overlay
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=False, color='blue', stat='density', label='Histogram', alpha=0.6)


# Overlay normal distribution
mean_residuals = np.mean(residuals)
std_residuals = np.std(residuals)
xmin, xmax = plt.xlim()  # Get the x limits
x = np.linspace(xmin, xmax, 100)  # Generate x values
p = norm.pdf(x, mean_residuals, std_residuals)  # Normal distribution PDF
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')


# Overlay kernel density estimate (KDE)
sns.kdeplot(residuals, color='red', linewidth=2, label='KDE')


plt.title('Histogram of Residuals with Normal and KDE Overlay')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()


# (formal)
shapiro(mod2.resid)




# -----------------------------------------------------------------------------------------------------------


## ***** NEO *********


# INFLUENTIAL VALUES
mod2 = smf.ols(formula='perform ~ gender + x07 + x09 + x11 + x16 + x20 + x21 + x28', data=xvar).fit()


# Print the summary of the model
print(mod2.summary())


# Create an OLSInfluence object
influence = smoi.OLSInfluence(mod2)


# Generate the summary table
summary_table = influence.summary_table()


# Print the summary table
print(summary_table)


# Get Cook's distance
cooks_d = influence.cooks_distance[0]
cutoff_cooks = 4 / len(mod2.resid)  # Typical cutoff for Cook's distance


# Get leverage (hat values)
leverage = influence.hat_matrix_diag
cutoff_leverage = 2 * (len(mod2.params) / len(mod2.resid))  # Cutoff for leverage


# Get DFFITS
dffits = influence.dffits[0]
cutoff_dffits = 2 * np.sqrt(len(mod2.params) / len(mod2.resid))  # Cutoff for DFFITS


# Get externally studentized residuals
student_resid = influence.resid_studentized_external
cutoff_student_resid = 2  # Generally, values beyond Â±2 are considered outliers


# Check for influential points based on cut-offs
influential_points = {
    'Cook\'s Distance': np.where(cooks_d > cutoff_cooks),
    'Leverage': np.where(leverage > cutoff_leverage),
    'DFFITS': np.where(np.abs(dffits) > cutoff_dffits),
    'Studentized Residuals': np.where(np.abs(student_resid) > cutoff_student_resid)
}


# Print the summary table and the influential points
summary_table = influence.summary_table()
print(summary_table)


print("\nInfluential Points based on cut-offs:")
for measure, indices in influential_points.items():
    print(f"{measure}: {indices[0]}")  # Print the indices of influential points




my_inf_mat = np.column_stack((cooks_d, leverage, dffits, student_resid))


# Convert Boolean values to 1 (True) and 0 (False)
my_inf_mat = my_inf_mat.astype(int)


# Sum rows; if any row has a sum of 1 or more, it's influential
influential_rows = np.where(np.apply_along_axis(np.sum, 1, my_inf_mat) >= 1)[0]


# Display influential indices
print("Influential points:", influential_rows)


data_swim.iloc[influential_rows,:]


data_swim_drop = data_swim.drop(data_swim.index[influential_rows])

xvar = data_swim_drop.iloc[:,4:32]
xvar.insert(0,'perform',perform)
mod_new_fit = smf.ols(formula='perform ~ gender + x07 + x09 + x11 + x16 + x20 + x21 + x28', data=data_swim_drop).fit()
mod_new_fit.summary()






# HOMOSKEDASTICITY


# (GRAPHICAL TEST)




print(residuals)


# Extract the design matrix (exogenous variables), excluding the constant
X_values = mod_new_fit.model.exog[:, 1:]  # This will only have the independent variables
residuals = mod_new_fit.resid




# Plotting X values versus the residuals
plt.figure(figsize=(12, 6))


# Define the independent variables based on your model
independent_vars = ['gender', 'x07', 'x09', 'x11', 'x16', 'x20', 'x21', 'x28']

# Assuming X_values is your matrix of independent variables
for i, col in enumerate(independent_vars):
    plt.subplot(2, 4, i + 1)  # Create subplots (2x4 grid for 8 variables)
    plt.scatter(data_swim_drop[col], residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel(col)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs {col}')



plt.tight_layout()
plt.show()

fitted_values =mod_new_fit.fittedvalues

# Plotting fitted values versus residuals
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# (formal)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, spec_white


# Perform the Breusch-Pagan test
bp_results = het_breuschpagan(mod_new_fit.resid, mod_new_fit.model.exog)


# Check the length of the results before unpacking
if len(bp_results) == 4:
    test_statistic, p_value, f_statistic, p_value_f = bp_results
    print("Breusch-Pagan p-value:", p_value)
else:
    print("Breusch-Pagan Test Results:", bp_results)


# Perform the White test
white_results = spec_white(mod_new_fit.resid, mod_new_fit.model.exog)


# Check the length of the results before unpacking
if len(white_results) >= 2:
    test_statistic, p_value = white_results[:2]  # Unpack only the first two elements
    print("White test p-value:", p_value)
else:
    print("White Test Results:", white_results)




# Test for random person

coef=mod_new_fit.params 

# Example coefficients from your fitted model (these are hypothetical)
beta_0 = coef.iloc[0]   # Intercept
beta_1 = coef.iloc[1]   # Coefficient for gender
beta_2 = coef.iloc[2]  # Coefficient for x07
beta_3 = coef.iloc[3]  # Coefficient for x09
beta_4 = coef.iloc[4]  # Coefficient for x11
beta_5 = coef.iloc[5]  # Coefficient for x16
beta_6 = coef.iloc[6]  # Coefficient for x20
beta_7 = coef.iloc[7]  # Coefficient for x21
beta_8 = coef.iloc[8] # Coefficient for x28


# Define the input values
gender = 1  # Male
x07 = 20    # Body alignment and posture
x09 = 85    # Decision-making
x11 = 10    # Endurance
x16 = 15    # Lactate threshold
x20 = 25    # Muscle strength
x21 = 18    # Pattern recognition
x28 = 22    # Technical skills


# Calculate predicted performance
predicted_performance = (
    beta_0 +
    beta_1 * gender +
    beta_2 * x07 +
    beta_3 * x09 +
    beta_4 * x11 +
    beta_5 * x16 +
    beta_6 * x20 +
    beta_7 * x21 +
    beta_8 * x28
)


# Output the predicted performance
print(f"The predicted performance of the male swimmer is: {predicted_performance:.2f}")


# ----------------------------END---OF---PHASE3---------------------------------------------


# PHASE 4