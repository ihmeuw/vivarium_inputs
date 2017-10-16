TAB = " "*4


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


if __name__ == "__main__":
    output = make_etiologies([("example1", 123), ("example2", 35), ("example3", 57)])
    with open(file="etiology.py", mode="w") as f:
        f.write(output)