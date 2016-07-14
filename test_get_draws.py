# we'll want to move this function into the gbd_ms_functions script probably

from transmogrifier import gopher

def get_me_id_draws(gbd_field_id,me_id,location_id):
	'''Returns a dataframe of 1,000 draws

    	Parameters
    	----------
   	gbd_id_field
	me_id : int
	me_id takes same me_id values as are used for GBD
	
	location_id : int
        location_id takes same location_id values as are used for GBD
        
    	Returns
    	-------
    	df
	'''
	if 
	df = gopher.draws(gbd_ids={'modelable_entity_ids': [me_id]},location_ids=location_id,sex_ids=[1, 2],status='latest',source='epi',verbose=True)















