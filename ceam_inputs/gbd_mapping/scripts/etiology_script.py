TAB = " "*4

from db_queries import get_rei_metadata

def make_etiology(name, rid):
    # building inner script
    out = ""
    out += TAB + f"{name}=Etiology(\n"
    out += TAB*2 + f"name='{name}',\n"
    out += TAB*2 + f"gbd_id=rid({rid}),\n"
    out += TAB + f"),\n"
    return out


def make_etiologies(etiology_list):
    out = "etiology=Etiologies(\n"
    for name, rid in etiology_list:
        out += make_etiology(name, rid)
    out += ")"
    return out

def get_etiology_data():
    etiologies = get_rei_metadata(rei_set_id = 3, gbd_round_id= 4)
    etiologies = etiologies[etiologies['most_detailed'] == 1]
    return list(zip(etiologies.rei_name, etiologies.rei_id))


if __name__ == "__main__":
    output = make_etiologies([("example1", 123), ("example2", 35), ("example3", 57)])
    with open(file="etiology.py", mode="w") as f:
        f.write(output)
