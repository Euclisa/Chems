import json
import re



def __clean_solubility(sol_str, ab_map):
    sol_str = re.sub(r',;', '', sol_str)
    sol_str = re.sub(r'\s+', ' ', sol_str)
    match = re.findall(r'(\bi-|\b[a-zA-Z0-9]+\b)', sol_str, re.ASCII)
    sol_cats = {'insoluble', 'miscible', 'soluble', 'slightly soluble', 'very soluble'}
    cleaned = dict()
    curr_sol_cat = None
    for word in match:
        if word in ab_map:
            word = word.replace(word, ab_map[word])
            if word in sol_cats:
                curr_sol_cat = word
            else:
                if curr_sol_cat is None:
                    return None
                cleaned[word] = curr_sol_cat
    
    return cleaned


def __clean_physical_form(phys_str, ab_map):
    phys_str = re.sub(r'\s+', ' ', phys_str)
    match = re.findall(r'(\bi-|\b[a-zA-Z0-9]+\b|\W)', phys_str, re.ASCII)
    cleaned = ''
    for word in match:
        if word in ab_map:
            word = word.replace(word, ab_map[word])
        cleaned += word
    
    return cleaned


def __check_approx(string):
    return True if re.search(r'[\u2248<>]', string) else False

def __clean_mp(mp_str):
    dec = 'dec' in mp_str
    approx = __check_approx(mp_str)
    value = __clean_float_value(mp_str)
    cleaned = {'decomposes': dec, 'value': value, 'approx': approx}

    return cleaned if dec or (value is not None) else None


def __clean_bp(bp_str):
    sub = 'sub' in bp_str or 'sp' in bp_str
    dec = 'dec' in bp_str
    approx = __check_approx(bp_str)
    value = __clean_float_value(bp_str)
    cleaned = {'sublimes': sub, 'decomposes': dec, 'value': value, 'approx': approx}

    return cleaned if dec or sub or (value is not None) else None


def __clean_float_value(val_str):
    value_match = re.findall(r'-?\d+(?:\.\d+)?', val_str)
    if len(value_match) != 1:
        value = None
    else:
        value = float(value_match[0])
    
    return value


def __clean_unicode(string):
    if not string:
        return None

    unicode_map = {
        '\ufb02': 'fl',
        '\ufb01': 'fi',
        '\u2019': "'",
        '\u2013': '-'
    }

    return ''.join(map(lambda c: unicode_map[c] if c in unicode_map else c, string))


def __clean_flash_point(fp_str):
    fp_str = __clean_unicode(fp_str)
    approx = __check_approx(fp_str)
    value = __clean_float_value(fp_str)
    cleaned = {'approx': approx, 'value': value}

    return cleaned if value is not None else None


def __clean_flash_limits(fl_str):
    fl_str = __clean_unicode(fl_str)
    fl_str = re.sub(r'\s+', '', fl_str)

    return fl_str


def __clean_ignition_temp(it_str):
    approx = __check_approx(it_str)
    value = __clean_float_value(it_str)
    cleaned = {'approx': approx, 'value': value}

    return cleaned if value is not None else None


def clean_organic_constants(input_fn, abbreviations_fn, output_fn=None):
    if output_fn is None:
        output_fn = input_fn

    with open(input_fn) as f:
        entries = [json.loads(x) for x in f.read().strip().split('\n')]

    with open(abbreviations_fn) as f:
        ab_map = [x.split('|') for x in f.read().strip().split('\n')]
        ab_map = dict(ab_map)

    for entry in entries:
        if not entry['name']:
            continue

        if entry['physical_form']:
            entry['physical_form'] = __clean_physical_form(entry['physical_form'], ab_map)
        
        if entry['solubility']:
            entry['solubility'] = __clean_solubility(entry['solubility'], ab_map)
        
        if entry['mp']:
            entry['mp'] = __clean_mp(entry['mp'])

        if entry['bp']:
            entry['bp'] = __clean_bp(entry['bp'])
        
        if entry['density']:
            entry['density'] = __clean_float_value(entry['density'])
        
        if entry['refractive_index']:
            entry['refractive_index'] = __clean_float_value(entry['refractive_index'])

        entry['name'] = __clean_unicode(entry['name'])
        entry['synonym'] = __clean_unicode(entry['synonym'])

        entry.pop('ind')


    with open(output_fn, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def clean_inorganic_constants(input_fn, abbreviations_fn, output_fn=None):
    if output_fn is None:
        output_fn = input_fn

    with open(input_fn) as f:
        entries = [json.loads(x) for x in f.read().strip().split('\n')]

    with open(abbreviations_fn) as f:
        ab_map = [x.split('|') for x in f.read().strip().split('\n')]
        ab_map = dict(ab_map)

    for entry in entries:
        if not entry['name']:
            continue

        if entry['physical_form']:
            entry['physical_form'] = __clean_physical_form(entry['physical_form'], ab_map)
        
        if entry['solubility']:
            entry['solubility'] = __clean_solubility(entry['solubility'], ab_map)
        
        if entry['mp']:
            entry['mp'] = __clean_mp(entry['mp'])

        if entry['bp']:
            entry['bp'] = __clean_bp(entry['bp'])
        
        if entry['density']:
            entry['density'] = __clean_float_value(entry['density'])
        
        if entry['solubility_aq']:
            entry['solubility_aq'] = __clean_float_value(entry['solubility_aq'])

        entry['name'] = __clean_unicode(entry['name'])

        entry.pop('ind')


    with open(output_fn, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def clean_flammability(input_fn, output_fn=None):
    if output_fn is None:
        output_fn = input_fn

    with open(input_fn) as f:
        entries = [json.loads(x) for x in f.read().strip().split('\n')]
    
    for entry in entries:
        if entry['flash_point']:
            entry['flash_point'] = __clean_flash_point(entry['flash_point'])
        
        if entry['flash_limits']:
            entry['flash_limits'] = __clean_flash_limits(entry['flash_limits'])
        
        if entry['ignition_temp']:
            entry['ignition_temp'] = __clean_ignition_temp(entry['ignition_temp'])

        entry['name'] = __clean_unicode(entry['name'])
    
    with open(output_fn, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


clean_organic_constants('data/crc_handbook/organic_constants.jsonl', 'data/crc_handbook/organic_abbreviations_map.txt')
clean_inorganic_constants('data/crc_handbook/inorganic_constants.jsonl', 'data/crc_handbook/inorganic_abbreviations_map.txt')
clean_flammability('data/crc_handbook/flammability.jsonl')