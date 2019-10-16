# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: s2ds
#     language: python
#     name: s2ds
# ---

# # Evaluate model performance at the aggregate level

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
from scipy.stats import binom
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
# -

# ## Load test data with predictions

# ## Overall accuracy

df_test = pd.read_pickle('customer_value.pkl')
df_test.head(10)

# ## Overall accuracy

# +
conv_rate_true = df_test['is_converted'].mean()
conv_rate_pred = df_test['predicted_conversion'].mean()
value_true = df_test['revenue'].mean()
value_pred = df_test['predicted_value'].mean()


conv_accuracy_err = (conv_rate_pred / conv_rate_true - 1.) * 100.
value_accuracy_err = (value_pred / value_true - 1.) * 100.
print("Accuracy error in conversion rate: {:.2f}%".format(conv_accuracy_err))
print("Accuracy error in customer value: {:.2f}%".format(value_accuracy_err))


# -

# The model sligtly underpredicts the conversion rate and customer value, but accurate to within 5%.
#
# This is partly due to seasonality variations in the conversion rate, since the period used for the beta calibration had a sligthly lower conversion rate, than that of the test period.

# ## Aggregate results

def aggregate_stats(df, on):
    #observed
    agg = {'is_converted': [
                ('n_visitors', 'count'),
                ('conversion_rate_true', lambda x: np.mean(x) * 100.),
                ],
           'revenue': [
                ('revenue_per_visitor_true', 'mean'),
                ],
           'predicted_conversion': [
                ('conversion_rate_pred', lambda x: np.mean(x) * 100.),
                ],
           'predicted_value': [
                ('revenue_per_visitor_pred', 'mean'),
                ],
          }
    df = df_test.groupby(on).agg(agg)
    df = df.droplevel(level=0, axis=1)
    
    df.insert(1, 'percentage_of_traffic', 100 * df['n_visitors'] / float(df['n_visitors'].sum()))
    
    return df


# ### per shop

df_shop = aggregate_stats(df_test, 'shop')
df_shop = df_shop.sort_values('n_visitors', ascending=False).reset_index()
df_shop

# ### per landing product

df_product = aggregate_stats(df_test, 'landing_product_type')
df_product = df_product.sort_values('n_visitors', ascending=False).reset_index()
df_product

# ### per keyword 

# +
df_keyword = df_test[df_test['medium'].str.contains('cpc')]
df_keyword = aggregate_stats(df_keyword, 'keyword')
df_keyword = df_keyword.sort_values('n_visitors', ascending=False).reset_index()

# only keep fequently used
df_keyword = df_keyword[(df_keyword['n_visitors'] > 2000) & ~df_keyword['keyword'].isin(['(not set)', '(not provided)'])] 
df_keyword
# -

# ## Is the model accurate when divided into subcategories?

# +
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4), dpi=200,
                                   tight_layout=True)

sizes = (20, 150)

s = sns.scatterplot(x='revenue_per_visitor_true', y='revenue_per_visitor_pred', 
                    size='n_visitors', hue='shop', data=df_shop,
                    ax=ax1, legend=False, sizes=sizes)

s = sns.scatterplot(x='revenue_per_visitor_true', y='revenue_per_visitor_pred',
                    size='n_visitors', hue='landing_product_type', data=df_product,
                    ax=ax2, legend=False, sizes=sizes)

s = sns.scatterplot(x='revenue_per_visitor_true', y='revenue_per_visitor_pred',
                    size='n_visitors', hue='keyword', data=df_keyword,
                    ax=ax3, legend=False, sizes=sizes)

for ax in [ax1, ax2, ax3]:
    ax.set_aspect(1)
    ax.set_xlabel("Statistical average vistor value [GBP]")
    ax.set_ylabel("Predicted average visitor value [GBP]")
    
ax1.plot([0, 5], [0, 5], ':', c='k', zorder=-1)
ax2.plot([0, 6], [0, 6], ':', c='k', zorder=-1)
ax3.plot([0, 9], [0, 9], ':', c='k', zorder=-1)

ax1.set_title('Shop')
ax2.set_title('Landing product type')
ax3.set_title('Keyword')

plt.show()
# -

# Accuracy of the model is still retained when divided by shop, product type and keyword.
#
# N.B. for a small number of visitors, there is considerable uncertaintly in both axes

# ## How precise is the two-step model vs the statistical method?

# Considering only the UK shop (with 360000 visitors), we can get a very good estimate "ground truth"

# +
df_test_gb = df_test[df_test['shop'] == 'gb']

conversion_rate_true = df_test_gb.is_converted.mean()
value_true = df_test_gb.revenue.mean()

print("Ground truth conversion rate: {:.2f}%".format(conversion_rate_true * 100.))
print("Ground truth visitor value: {:.2f} GBP".format(value_true))
# -

# Statistically the we expect the number of observed conversions to follow a binomial distribution.
#
# So the standard deviation in the number of observed converions is $\sqrt{N p (1-p)}$, where $p$ is the true conversion probability, and $N$ is the total number of visitors.
#
# Therefore the error in the estimated probability is $\displaystyle \sqrt{\frac{p (1-p)}{N}}$.
#
# I.e. error in the probability scales proportionally with $\displaystyle \propto \frac{1}{\sqrt{N}}$.

# Calculately the error in the two-step model cannot be done analytically.
# However, if we take a random sub-sample (for example of only 1000 visitor) we can derive a predicted conversion rate from the model.
#
# Repeating this a large number times for different sub-samples we can can measure the scatter in the model prediciton. This should represent the upper-bounds on the model error.
#
# We can also calculate the error on the visitor value the same way (for both the statistical and two-step model methods).

# +
#for sample sizes ranging between 1 and 100000
n_sample = np.logspace(0,5,11).astype(int)

#theoretical error from binomial distribution
conv_samp_err_theo = binom.std(n_sample, conversion_rate_true) / n_sample.astype(float)

#empty arrays for storing results
conv_samp_err = []
conv_pred_err = []
value_samp_err = [] 
value_pred_err = []

random_state = np.random.RandomState(0)
for n in n_sample:
    #for smaller sample sizes perform more Monte Carlo iterations
    n_mc = int(np.clip(1e5 / n, 100, 10000))
    
    conv_samp_mc = []
    conv_pred_mc = []
    value_samp_mc = []
    value_pred_mc = []
    for _ in range(n_mc):
        #compute sub-sample estimates
        df = df_test_gb.sample(n, replace=True, random_state=random_state)
        conv_samp_mc.append(df['is_converted'].mean())
        conv_pred_mc.append(df['predicted_conversion'].mean())
        value_samp_mc.append(df['revenue'].mean())
        value_pred_mc.append(df['predicted_value'].mean())
        
    #compute scatter in estimates
    conv_samp_err.append(np.std(conv_samp_mc, ddof=1))
    conv_pred_err.append(np.std(conv_pred_mc, ddof=1))
    value_samp_err.append(np.std(value_samp_mc, ddof=1))
    value_pred_err.append(np.std(value_pred_mc, ddof=1))
    
#convert to numpy arrays
conv_samp_err = np.array(conv_samp_err)
conv_pred_err = np.array(conv_pred_err)
value_samp_err = np.array(value_samp_err)
value_pred_err = np.array(value_pred_err)
# -

# Compare sampling matches theoretical expectation, not guarenteed if there is not a good single "ground-truth"

print(conv_samp_err / conv_samp_err_theo)

# Good enough

# +
fig, (ax1, ax2),  = plt.subplots(1, 2, figsize=(8,4), dpi=300, tight_layout=True)

ax1.plot(n_sample, conv_samp_err * 100, label='Statistical method')
ax1.plot(n_sample, conv_pred_err * 100, label='Two-step model')

ax2.plot(n_sample, value_samp_err, label='Statistical method')
ax2.plot(n_sample, value_pred_err, label='Two-step model')

ax2.legend()

ax1.set_xscale('log')
ax1.set_xlabel("# visitors")
ax1.set_ylabel("Error on conversion rate [% points]")

ax1.set_xlim([10,100000])
ax1.set_ylim([0,4])

ax2.set_xscale('log')
ax2.set_xlabel("# visitors")
ax2.set_ylabel("Error on visitor value")

ax2.set_xlim([10,100000])
ax2.set_ylim([0, 25])

plt.show()
# -

# Both methods have the same $\frac{1}{\sqrt{n}}$ scaling in precision. But for a fixed number of visitors the model out-performs the statistical method.

# +
conv_rate_improvement = np.mean((1 - (conv_pred_err / conv_samp_err)) * 100)
value_improvement = np.mean((1 - (value_pred_err / value_samp_err)) * 100)

print("Model reduces error on conversion rate by ~{:.0f}%".format(conv_rate_improvement))
print("Model reduces error on visitor value by ~{:.0f}%".format(value_improvement))
# -

# Given the $\frac{1}{\sqrt{n}}$ scaling, to reduce the error in the statistical method by $X\%$, one would need to increase the $n$ by a factor
#
# $\displaystyle \frac{n_{\rm new}}{n} \approx \left(1 - \frac{X}{100}\right)^{-2}$
#
# if the error in the visitor value scales similarly, a 65% improvement in error equates to needing 8x or 9x less data

# However the improvement in data volumn can be obtained by iterpolating between the two curves

print(n_sample)
print(np.interp(value_pred_err, value_samp_err[::-1], n_sample[::-1])  / n_sample)

# Ignoring the last few (where the interpolation fails), we can see that indeed there is roughly a 9x improvement
