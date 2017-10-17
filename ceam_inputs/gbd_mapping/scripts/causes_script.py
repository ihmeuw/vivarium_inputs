TAB = " "*4
TEXTWIDTH = 120


def make_cause(name, cid, seq_names=None, etiol_names=None):
    # building inner script
    out = ""
    out += TAB + f"{name}=Cause(\n"
    out += TAB*2 + f"name='{name}',\n"
    out += TAB*2 + f"gbd_id=UNKNOWN, #cid({cid}),\n"

    # Sequelae
    if seq_names:
        seq_field = TAB*2 + "sequela=("
        offset = len(seq_field)

        out += seq_field
        char_count = offset

        if isinstance(seq_names, tuple):
            for each in seq_names:
                seq_name = f"sequela.{each}, "
                seq_name_len = len(seq_name)

                if char_count == offset:
                    out += seq_name
                    char_count += seq_name_len
                elif char_count > TEXTWIDTH:
                    out += "\n" + " "*offset + seq_name
                    char_count = offset + seq_name_len
                else:
                    out += seq_name
                    char_count += seq_name_len

            out += "),"

        else:
            out += f"sequela.{seq_names}),"

    if etiol_names:
        etiol_field = "\n" + TAB*2 + "etiologies=("
        offset = len(etiol_field)

        out += etiol_field
        char_count = offset

        if isinstance(etiol_names, tuple):
            for each in etiol_names:
                etiol_name = f"etiology.{each}, "
                etiol_name_len = len(etiol_name)

                if char_count == offset:
                    out += etiol_name
                    char_count += etiol_name_len
                elif char_count > TEXTWIDTH:
                    out += "\n" + " "*offset + etiol_name
                    char_count = offset + etiol_name_len
                else:
                    out += etiol_name
                    char_count += etiol_name_len
        else:
            out += f"etiology.{etiol_names}, "

        out += "),"

    out += "\n" + TAB + "),"
    return out


def make_causes(causes_list):
    out = "causes=Causes(\n"
    for name, cid, seq_id, etiol_id in causes_list:
        out += make_cause(name, cid, seq_id, etiol_id)
        out += "\n"
    out += ")"
    return out


if __name__ == "__main__":
    output = make_causes([("diarrhea", 202, "bananas", ("etiol1", "etiol2", "etiol3", "etiol4", "etiol5", "etiol1",
                                                        "etiol2", "etiol3", "etiol4", "etiol5",
                                                        "etiol1", "etiol2", "etiol3", "etiol4", "etiol5",)),
             ("LRI", 231, ("seq2", "seq3123123123123123123123123123", "seq4", "seq5", "seq6", "seq8",
                           "seq1231231231231231231231231231232", "seq3", "seq4", "seq5", "seq6", "seq8", "seq2", "seq4",
                           "seq5", "seq6", "seq8", "seq3", "seq4", "seq5", "seq6", "seq8", ), "example_1")])

    with open(file="causes.py", mode="w") as f:
        f.write(output)

## Won't work w/ only 3 arguments. Needs etiol_id to be None if there is no etiolog data