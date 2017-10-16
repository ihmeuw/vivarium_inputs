TAB = " "*4


def make_sequela(name, sid, mei_id):
    # building script
    out = ""
    out += TAB + f"{name}=Sequela(\n"
    out += TAB*2 + f"name='{name}',\n"
    out += TAB*2 + f"gbd_id=sid({sid}),\n"
    out += TAB*2 + f"dismod_id=meid({mei_id}),\n"
    out += TAB + f"),\n"
    return out


def make_sequelae(sequela_list):
    out = "sequela=Sequelae(\n"
    for name, sid, mei_id in sequela_list:
        out += make_sequela(name, sid, mei_id)
    out += ")"
    return out


if __name__ == "__main__":
    output = make_sequelae([("example1", 12, 123), ("example2", 34, 35), ("example3", 56, 57)])
    with open(file="sequela.py", mode="w") as f:
        f.write(output)