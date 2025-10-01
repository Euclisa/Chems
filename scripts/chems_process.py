import json
import re

def extract_pubchem(in_fn, out_fn):
    fields_to_keep = ['cid', 'cmpdname', 'cmpdsynonym', 'mf', 'mw', 'complexity', 'smiles', 'inchi', 'inchikey', 'charge']
    curr_chem = None
    i = 0
    with open(in_fn) as f_in, open(out_fn, 'w') as f_out:
        for line in f_in:
            for field in fields_to_keep:
                if f'"{field}":' in line:
                    if field == 'cid':
                        if curr_chem is not None:
                            if 'cmpdsynonym' in curr_chem:
                                curr_chem['charge'] = int(curr_chem['charge'])
                                curr_chem['cid'] = int(curr_chem['cid'])
                                curr_chem['mw'] = float(curr_chem['mw'])
                                curr_chem['complexity'] = float(curr_chem['complexity'])
                                if not isinstance(curr_chem['cmpdsynonym'], list):
                                    curr_chem['cmpdsynonym'] = [curr_chem['cmpdsynonym']]
                                
                                f_out.write(json.dumps(curr_chem) + '\n')

                                i += 1
                                if i % 1000 == 0:
                                    print(i)
                        curr_chem = dict()
                    curr_chem[field] = json.loads('{' + line.strip().strip(',') + '}')[field]


def filter_sort_chems(chems_fn):
    with open(chems_fn) as f:
        chems = [json.loads(x) for x in f.read().strip().split('\n')]
    
    new_chems = dict()
    for chem in chems:
        if chem['charge'] == 0 and chem['complexity'] < 300 and ';' not in chem['cmpdname'] and not re.search(r'\d{3,}', chem['cmpdname']):
            new_chems[chem['cid']] = chem

    new_chems = list(new_chems.values())
    new_chems.sort(key=lambda x: x['complexity'])

    with open(chems_fn, 'w') as f:
        for chem in new_chems:
            f.write(json.dumps(chem) + '\n')
    
    print(f"New chems list size: {len(new_chems)}")

extract_pubchem('pubchem.json', 'chems.jsonl')
filter_sort_chems('chems.jsonl')
                                


