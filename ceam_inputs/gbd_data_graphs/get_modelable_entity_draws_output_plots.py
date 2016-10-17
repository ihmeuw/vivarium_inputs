get_ipython().system(u'date ')

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
# from matplotlib.backends.backend_pdf import PdfPages  

# get_ipython().magic(u'matplotlib inline')
# sns.set_context('paper')
# sns.set_style('darkgrid')
# pd.set_option('max_rows',5)


# # Function Output Plots
# This notebook plots the outputs of the gbd_ms_functions. We need plots to ensure that our data match what is coming from DisMod (can check that the outputs match by using the epi viz tool -- http://epimodeling-web-p01.ihme.washington.edu/2015/). More importantly, for the causes/measures that we can't pull straight from DisMod (e.g. excess mortality of heart failure due to ihd), these plots will serve as a sanity check for the numbers which we are producing.

from ceam_inputs.gbd_ms_functions import *
from ceam_inputs.gbd_ms_auxiliary_functions import *


def plots(me_id, measure):

    # output a flipbook to a pdf
    out_path = '/share/costeffectiveness/CEAM/gbd_ms_function_output_plots/{m}_{ms}.pdf'.format(m= me_id, ms = measure)
    pp = PdfPages(out_path)

    data = get_modelable_entity_draws(180, 1990, 2010, measure, me_id)
    
    for sex_id in [1,2]:
        for year_id in [1990,2010]:

            df = data.query("sex_id == {s} and year_id == {y}".format(s=sex_id, y=year_id))

            # create cols for mean estimates and upper/lower conf. intervals
            df['lower'] = df[['draw_{i}'.format(i=i) 
                          for i in range(0,1000)]].quantile(0.025, axis=1)
            df['upper'] = df[['draw_{i}'.format(i=i) 
                          for i in range(0,1000)]].quantile(0.975, axis=1)
            df['mean'] = df[['draw_{i}'.format(i=i) 
                         for i in range(0,1000)]].mean(axis=1)
            
            plt.plot(df.age.values,df['mean'].values,'b-',label='mean with ui')
            plt.fill_between(df.age.values, df['lower'].values,
                             df['upper'].values, alpha=0.5, color = 'b')
            
            db_query = pd.read_csv("/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data/draws_for_location180_for_meid{}.csv".            format(me_id))
            db_query = get_age_from_age_group_id(db_query)
            
            db_query= db_query.query("year_id == {y} and sex_id == {s} and measure_id == {m}".            format(y=year_id, s=sex_id, m=measure))

            db_query['mean'] = db_query[['draw_{i}'.format(i=i) for i in range(0,1000)]].mean(axis=1)
            plt.plot(db_query.age.values, db_query['mean'], 'ro', label='db')
            

            plt.title('{m}_{ms}_{y}_{s}'.format(m = me_id, ms = measure, y = year_id, s = sex_id))
            plt.legend(loc = 9)
            plt.xlabel('Age (years)')
            plt.ylabel('{}'.format(measure))
            plt.xlim((20,80))

            pp.savefig()
            plt.clf()

    pp.close()

	
for i in [1814, 3233, 1817]:
    for x in [5, 6, 9]:
        plots(i, x)

