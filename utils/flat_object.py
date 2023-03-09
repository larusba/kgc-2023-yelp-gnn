import ast

def flat_object(obj, prefix=''):
    new_obj = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            sub_obj = flat_object(v, prefix=f'{prefix}{k}_')
            new_obj = {**new_obj, **sub_obj}
        elif isinstance(v, str):
            try:
                elem = ast.literal_eval(v)
                if isinstance(elem, dict):
                    sub_obj = flat_object(elem, prefix=f'{prefix}{k}_')
                    new_obj = {**new_obj, **sub_obj}
                else:
                    new_obj[prefix + k] = elem
            except:
                new_obj[prefix + k] = v
        else:
            new_obj[prefix + k] = v
    return new_obj