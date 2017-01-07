from transmogrifier.draw_ops import get_draws
import pandas as pd
import sys

draws = get_draws(gbd_id_field="modelable_entity_id", gbd_id=sys.argv[2], location_ids=sys.argv[1], source="epi", age_group_ids=list(range(2,22)), status=sys.argv[3])

draws.to_csv(sys.argv[4])
