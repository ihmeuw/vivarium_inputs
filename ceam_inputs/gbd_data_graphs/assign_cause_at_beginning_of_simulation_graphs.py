
# coding: utf-8

# In[1]:



# In[3]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set_context('paper')
sns.set_style('darkgrid')
pd.set_option('max_rows',5)
from matplotlib.backends.backend_pdf import PdfPages  


from ceam_inputs import generate_ceam_population
from ceam_inputs import get_prevalence
from ceam_inputs.gbd_ms_functions import assign_cause_at_beginning_of_simulation


# In[6]:

pop = generate_ceam_population(100000)


# In[94]:

number_of_distinct_pops = 15


# In[30]:

states = {'diarrhea' : get_prevalence(1181)}


# In[95]:

pop_w_causes = {}
for i in range(0, number_of_distinct_pops):
    pop_w_causes['{}'.format(i)] = assign_cause_at_beginning_of_simulation(pop, 180, 1990, states)


# In[118]:

# def prevalence_plot(me_id, disease, sex_id):

# specify an outpath for the graphs
#out_path = '/share/costeffectiveness/CEAM/gbd_ms_function_output_plots/{d}_prevalence_among_sex_id{s}.pdf'.format(d=disease, s=sex_id)

# pp = PdfPages(out_path)

me_id = 1181
sex_id = 1
disease = 'diarrhea'

prev_dict = {}

for i in range(0, number_of_distinct_pops):

    onesex = pop_w_causes['{}'.format(i)].query("sex_id == {}".format(sex_id))

    sick_df = onesex.loc[onesex.condition_state == disease]

    sick_df['count'] = 1
    sick_group = sick_df.groupby('age')[['count']].sum()

    total_pop = pop_w_causes['{}'.format(i)].query("sex_id == {}".format(sex_id))
    total_pop['count'] = 1
    total_pop = total_pop.groupby('age')[['count']].sum()

    prev_dict['{s}_{i}'.format(s=sex_id, i=i)] = sick_group.join(total_group, lsuffix='_sick', rsuffix='_total', how='outer')

    prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['sex_id'] = sex_id

    prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['prevalence_{}'.format(i)] = prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['count_sick'] /         prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['count_total']

    prev_dict['{s}_{i}'.format(s=sex_id, i=i)].reset_index(inplace=True)
    
merged = pd.merge(prev_dict['{}_0'.format(sex_id)], prev_dict['{}_1'.format(sex_id)], on =['age', 'sex_id'], how='outer')

for i in range(2, number_of_distinct_pops):
    merged = pd.merge(merged, prev_dict['{s}_{i}'.format(s = sex_id, i = i)], on=['age', 'sex_id'], how='outer')

# This block creates a list of all of the prevalence columns (e.g. ['prevalence_0', 'prevalence_1', ...])
# Need this list to calculate the mean, upper, and lower
prevalence_columns = []
for i in range(0, number_of_distinct_pops):
    prevalence_columns.append('prevalence_{i}'.format(i=i))

# Fill in the nulls with 0 since null at this point means no one at a given age has the disease of interest
merged = merged.fillna(0)    

merged['mean_prev'] = merged[prevalence_columns].mean(axis=1)
merged['upper'] = merged[prevalence_columns].quantile(0.975, axis=1)
merged['lower'] = merged[prevalence_columns].quantile(0.025, axis=1)

merged = merged[['age', 'sex_id'] + prevalence_columns + ['mean_prev', 'upper', 'lower']]

draw_prev = get_prevalence(me_id)
draw_prev = draw_prev.query('year == 1990')
draw_prev = draw_prev[['age', 'sex', 'prevalence']]

if sex_id == 1:
    s = 'Male'
else:
    s ='Female'
    
draw_prev = draw_prev.query("sex == '{}'".format(s))

plt.figure()

plt.plot(draw_prev.age.values, draw_prev.prevalence.values, 's', label ='prev from database (draw_0)', color='red')
plt.errorbar(merged.age.values, merged.mean_prev.values, yerr = [merged.mean_prev.values - merged.lower.values, 
                                                               merged.upper.values - merged.mean_prev.values], fmt='o', 
            label = 'mean prevalence in simulation w/ error bars')

plt.xlim((0,81))
plt.title("{d} prevalence among sex_id = {s}".format(d = disease, s = sex_id))
plt.xlabel('age')
plt.ylabel('prevalence')
plt.legend()

# pp.savefig()
# plt.clf()

# pp.close()


# In[ ]:

for sex_id in [1, 2]:
    for key, value in states.items():
        prevalence_plot(value, key, sex_id)

