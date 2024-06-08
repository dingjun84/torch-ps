import json
from collections import OrderedDict


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())