TAB = " "*4
TEXTWIDTH = 120


def make_risk_factor(name, rid, distribution, levels, tmred, exposure_parameters, causes_list):
    # building inner script
    out = ""
    out += TAB + f"{name}=Risk(\n"
    out += TAB*2 + f"name='{name}',\n"
    out += TAB*2 + f"gbd_id=UNKNOWN, #rid({rid}),\n"
    out += TAB*2 + f"distribution='{distribution}',\n"

    if distribution == "Nothing":
        # levels script - categorical
        levels_len = len(levels)
        out += TAB*2 + "levels=Levels(\n"

        for each in range(levels_len):
            out += TAB*3 + f"cat{each}='{levels[each-1]}',\n"

        out += TAB*2 + "),\n"

    else:
        # tmred & exposure parameters - continuous
        # Assuming data can be in tuple of tuples format

        out += TAB*2 + "tmred=Tmred(\n"
        for name, value in tmred:
            if name == 'distribution':
                out += TAB*3 + f"{name}='{value}',\n"
            else:
                out += TAB*3 + f"{name}={value},\n"
        out += TAB*2 + "),\n"

        out += TAB*2 + "exposure_parameters=ExposureParameters(\n"
        for name, value in exposure_parameters:
            out += TAB*3 + f"{name}={value},\n"
        out += TAB*2 + "),\n"

    # Affected_causes
    out += TAB*2 + "affected_causes(\n"
    for each in causes_list:
        out += TAB*3 + f"causes.{each},\n"
    out += TAB*2 + "),\n"

    out += TAB + "),"

    return out


def make_risk_factors(riskfactor_data):
    out = "risk_factors=Risks(\n"
    for name, rid, distribution, levels, tmred, exposure_parameters, causes_list in riskfactor_data:
        out += make_risk_factor(name, rid, distribution, levels, tmred, exposure_parameters, causes_list)
        out += "\n"
    out += ")"
    return out


one_categorical = ("unsafe_water_source", 83, "Nothing", ("unimproved and untreated", "unimproved and chlorinated",
                                                            "unimproved and filtered", "unimproved and untreated",
                                                            "improved and untreated", "improved and chlorinated",
                                                            "improved and filtered", "piped and untreated",
                                                            "piped and chlorinated", "piped and filtered"), None, None,
                   ('lower_respiratory_infections', 'tracheal_bronchus_and_lung_cancer'))


two_continuous = ("ambient_particulate_matter_pollution", 86, "unknown", None, (("distribution", "uniform"), ("min", 5), ("max",5000), ("inverted", True)),
                 (("scale", 1),("max_rr",'500'), ("min_rr", 7565.3654), ("min_val", 220), ("max_var", 130)),
                 ('lower_respiratory_infections', 'tracheal_bronchus_and_lung_cancer'))


if __name__ == "__main__":
    output = make_risk_factors((one_categorical, two_continuous))

    with open(file="riskfactors.py", mode="w") as f:
        f.write(output)


