TAB = " "*4
TEXTWIDTH = 120

#add healthstate field
#another class - after dismod_id
# healthstate= Healthstate(


def make_sequela(name, sid, mei_id, hs_name, hid):
    # building script
    out = ""
    out += TAB + f"{name}=Sequela(\n"
    out += TAB*2 + f"name='{name}',\n"
    out += TAB*2 + f"gbd_id=sid({sid}),\n"
    out += TAB*2 + f"dismod_id=meid({mei_id}),\n"
    out += TAB*2 + f"healthstate=Healthstate(\n"
    out += TAB*3 + f"name='{hs_name}',\n"
    out += TAB*3 + f"id=hid({hid}),\n"
    out += TAB*2 + f"),\n"
    out += TAB + f"),\n"
    return out


def make_sequelae(sequela_list):
    out = "sequela=Sequelae(\n"
    for name, sid, mei_id, hs_name, hid in sequela_list:
        out += make_sequela(name, sid, mei_id, hs_name, hid)
    out += ")"
    return out


if __name__ == "__main__":
    output = make_sequelae([("example1", 12, 123, "example_hs", 1111), ("example2", 34, 35, "example_hs2", 2222),
                            ("example3", 56, 57, "example_hs3", 33333)])

    with open(file="sequela.py", mode="w") as f:
        f.write(output)
