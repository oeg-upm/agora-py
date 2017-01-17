import base64


def encode_rdict(rd, parent_item=None):
    if parent_item:
        rd["$parent"] = parent_item

    sorted_keys = sorted(rd.keys())
    sorted_fields = []
    for k in sorted_keys:
        sorted_fields.append("{}: {}".format(str(k), str(rd[k])))
    str_rd = '{' + ','.join(sorted_fields) + '}'
    return base64.b64encode(str_rd)
