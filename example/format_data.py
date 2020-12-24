# %%
import json

season = 1

with open(f"json/MakePuzz_Season{season}_original.json") as f:
    data = json.load(f)["__collections__"]["json"]
    for key in data:
        data[key].pop("__collections__")

# %%
def remove_bug(data):
    """
    Parameters
    ----------
    data : dict
    """
    from copy import deepcopy
    new_data = deepcopy(data)
    for fname in data:
        try:
            if new_data[fname]["bug"]:
                new_data.pop(fname)
        except KeyError:
            print("KeyError: " + fname)
            new_data.pop(fname)
    return new_data

def parse_fname(fname):
    import re
    pattern = '_w([\d]+)_h([\d]+)_s([\d]+)_e([\d]+)'
    parsed = re.split(pattern, fname)
    ret = {"name": parsed[0],
           "width": int(parsed[1]), 
           "height": int(parsed[2]), 
           "seed": int(parsed[3]), 
           "epoch": int(parsed[4])}
    return ret

def validate_seed(data):
    from copy import deepcopy
    new_data = deepcopy(data)
    for fname in data:
        parsed = parse_fname(fname)
        if new_data[fname]["seed"] != parsed["seed"]:
            new_data[fname]["seed"] = parsed["seed"]
            print("seed replaced: ", fname)
    return new_data

# %%
odata = remove_bug(data)
odata = validate_seed(odata)
#%%
with open(f"json/MakePuzz_Season{season}.json", "w") as f:
    json.dump(odata, f)
# %%
