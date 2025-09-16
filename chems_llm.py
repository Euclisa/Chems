from openai import OpenAI
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import pubchempy as pcp


class ChemsLLM:
    def __init__(self, data_dir, api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        self.completion_tokens_total = 0
        self.input_tokens_total = 0
        self.tokens_total_lock = threading.Lock()
        
        self.print_lock = threading.Lock()

        self.gpt_oss = "openai/gpt-oss-120b"
        self.qwen = "qwen/qwen3-235b-a22b-thinking-2507"
        self.grok = "x-ai/grok-3-mini"
        self.gemeni = "google/gemini-2.5-flash-lite"

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.raw_reactions_fn = os.path.join(self.data_dir, "raw_reactions.jsonl")
        self.raw_reactions_verdict_fn = os.path.join(self.data_dir, "raw_reactions_verdict.jsonl")
        self.raw_reactions_staged_fn = os.path.join(self.data_dir, "raw_reactions_staged.jsonl")
        self.reactions_parsed_fn = os.path.join(self.data_dir, "reactions_parsed.jsonl")
        self.unmapped_names_fn = os.path.join(self.data_dir, "unmapped_names.txt")
        self.unmapped_names_blacklisted_fn = os.path.join(self.data_dir, "unmapped_names_blacklisted.txt")
        self.chems_fn = os.path.join(self.data_dir, "chems.jsonl")



    def log(self, message=""):
        with self.print_lock:
            print(message)
    
    def __fetch_llm_answer(self, messages, model):
        completion = self.client.chat.completions.create(model=model, messages=messages)

        with self.tokens_total_lock:
            self.input_tokens_total += completion.usage.prompt_tokens
            self.completion_tokens_total += completion.usage.completion_tokens

        return completion.choices[0].message.content
    

    def __get_processed_entries(self, out_fn, id):
        processed = set()

        if os.path.exists(out_fn):
            with open(out_fn) as f:
                entries = f.read().strip().split('\n')
            
            for entry in entries:
                if entry:
                    processed.add(json.loads(entry)[id])
        
        return processed
    

    def __get_reactions_from_response(self, response: str):
        reactions = []
        for line in response.split('\n'):
            if '->' in line:
                reactions.append(line)
        
        return reactions
    
    
    def __fetch_raw_reactions(self, chem):
        try:
            chem_name = chem['cmpdname']

            instruct = \
            f"Please, provide a comprehensive list of documented chemical reactions involving {chem_name}, where it appears as either a reactant or a product. " \
            "Write the reactions as schemes using only '->' and '+' symbols. Use the full chemical names of substances instead of formulas or generic terms. " \
            "Do not include balancing coefficients, comments, or any markup - only the reaction schemes themselves. " \
            "If no such substance exists or no documented reactions are available, return 'None'."

            instruct_revalidate = \
            "Please, review the provided reactions. Identify any erroneous reactions and correct them where possible. Return revised reactions list that comply with the initial requirements."

            instruct_validate = \
            "You will be given a list of unbalanced chemical reaction schemes. " \
            "For each scheme, determine if the reaction is chemically possible and whether the listed reactants and products are correct. " \
            "If the reaction is valid, output only 'Valid'. If it is not valid, output only 'Invalid'. " \
            "Print one result per line and do not include any additional text."


            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instruct}
            ]

            models_to_try = [self.gpt_oss, self.grok]
            
            for curr_model in models_to_try:
                model = curr_model
                response = self.__fetch_llm_answer(messages, model)
                reactions = self.__get_reactions_from_response(response)
                if reactions:
                    break
            else:
                self.log(f"Failed to fetch reactions for '{chem_name}'")
                return None
            
            #self.log(f"Got {len(reactions)} initial reactions for {chem_name}")
            
            response = '\n'.join(reactions)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": instruct_revalidate})

            response = self.__fetch_llm_answer(messages, model)
            reactions_revalid = self.__get_reactions_from_response(response)
            if not reactions_revalid:
                self.log(f"Failed to fetch revalidated reactions for '{chem_name}'. Assuming all are valid...")
                reactions_revalid = reactions
            
            self.log(f"Got {len(reactions)} reactions for {chem_name} with '{model}'; CTT: {self.completion_tokens_total}")
        
        except Exception as e:
            self.log(f"Exception in '__fetch_raw_reactions': {e}")
            return None
        
        return {'cid': chem['cid'], 'reactions': reactions}



    def get_raw_reactions(self, max_workers=1):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed = self.__get_processed_entries(self.raw_reactions_fn, 'cid')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in chems:
                if chem['cid'] not in processed:
                    futures.append(executor.submit(self.__fetch_raw_reactions, chem))
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
    

    def __get_verdicts_bool_from_response(self, response: str):
        verdicts = []
        for line in response.split('\n'):
            if 'invalid' in line.lower():
                verdicts.append(False)
            elif 'valid' in line.lower():
                verdicts.append(True)
        
        return verdicts
    

    def __validate_raw_reactions(self, reactions, valid_cnt):
        try:
            instruct_validate = \
            "You will be given a list of unbalanced chemical reaction schemes. " \
            "For each scheme, determine if the reaction is chemically possible and whether the listed reactants and products are correct. " \
            "If the reaction is valid, output only 'Valid'. If it is not valid, output only 'Invalid'. " \
            "Print one result per line and do not include any additional text."

            reactions_str = '\n'.join([f"{i+1}. {react['reaction']}" for i, react in enumerate(reactions)])

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{instruct_validate}\n{reactions_str}"}
            ]

            models_to_try = [self.gpt_oss, self.grok]
            model_i = 0

            valid_i = 0
            mistakes_cnt = 0
            valid_ratio = [0.0 for i in range(len(reactions))]
            while valid_i < valid_cnt:
                response = self.__fetch_llm_answer(messages, models_to_try[model_i])
                verdicts = self.__get_verdicts_bool_from_response(response)
                if len(verdicts) != len(reactions):
                    self.log(f"Mistake: {len(verdicts)} != {len(reactions)}: '{response}'")
                    mistakes_cnt += 1
                    if mistakes_cnt > 1:
                        # switch to fallback if main model refuses or fails to validate
                        model_i += 1
                        if model_i == len(models_to_try):
                            self.log(f"Failed to validate:\n{reactions_str}")
                            return None
                        # 2 more validation tries for fallback model
                        valid_cnt += 2
                        self.log(f"Falling to {models_to_try[model_i]} on try {valid_i}")
                    continue
                
                for i, verd in enumerate(verdicts):
                    valid_ratio[i] += float(verd)
                
                valid_i += 1

            for i, ratio in enumerate(valid_ratio):
                valid_ratio[i] /= valid_cnt

            valid_reacts_cnt = 0
            verdict_reactions = []
            for i, react in enumerate(reactions):
                react['valid'] = valid_ratio[i] > 0.5
                verdict_reactions.append(react)

                if react['valid']:
                    valid_reacts_cnt += 1
            
            self.log(f"{valid_reacts_cnt}/{len(reactions)} are valid ('{models_to_try[model_i]}'); CTT: {self.completion_tokens_total}")
            
            return verdict_reactions
        
        except Exception as e:
            self.log(f"Exception in '__validate_raw_reaction': {e}")
            return None
    

    def validate_raw_reactions(self, max_workers=1):
        with open(self.raw_reactions_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed = self.__get_processed_entries(self.raw_reactions_verdict_fn, 'reaction')
        
        reactions = []

        for entry in entries:
            cid = entry['cid']
            reactions_curr = entry['reactions']
            for react in reactions_curr:
                if react not in processed:
                    reactions.append({'cid': cid, 'reaction': react})
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.raw_reactions_verdict_fn, 'a') as f_out:
            futures = []
            i = 0
            while i < len(reactions):
                futures.append(executor.submit(self.__validate_raw_reactions, reactions[i:i+10], 5))
                i += 10
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                for react in res:
                    f_out.write(json.dumps(react) + '\n')
                f_out.flush()
    
    def find_all_unicode_chars_in_raw_reactions(self):
        non_ascii = dict()
        with open(self.raw_reactions_verdict_fn) as f:
            for line in f:
                reaction = json.loads(line)['reaction']
                non_ascii_curr = [char for char in reaction if ord(char) > 127]
                for char in non_ascii_curr:
                    if char not in non_ascii:
                        non_ascii[char] = 0
                    non_ascii[char] += 1
        
        return non_ascii
    

    def __chem_name_to_ascii(self, chem_name):
        unicode_map = {
            '‑': '-',
            'α': 'alpha',
            'γ': 'gamma,',
            '–': '-'
        }
        chem_name_ascii = ""
        for char in chem_name:
            if not char.isascii():
                if char in unicode_map:
                    char = unicode_map[char]
                else:
                    char = ""
            chem_name_ascii += char
        
        return chem_name_ascii

    def __normalize_chem_name(self, chem_name):
        chem_name = chem_name.lower()
        chem_name = self.__chem_name_to_ascii(chem_name)
        chem_name = chem_name.replace('vapor', '')
        chem_name = chem_name.replace('dust', '')
        chem_name = chem_name.replace('elemental', '')
        chem_name = chem_name.replace('metal', '')
        chem_name = chem_name.replace('aqueus', '')
        chem_name = chem_name.strip()
        chem_name = chem_name.strip('`\'".,;:')
        chem_name = re.sub(r'\s+', ' ', chem_name.lower())
        chem_name = re.sub(r' \([^\d]+\)$', '', chem_name)
        chem_name = re.sub(r'^\d+ ', '', chem_name)

        return chem_name
    

    def __extract_name_cid_map(self):
        name_cid_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                entry = json.loads(line)
                cid = entry['cid']
                name_cid_map[self.__normalize_chem_name(entry['cmpdname'])] = cid
                synonyms = list(filter(lambda x: not re.search(r'\d{3,}', x), entry['cmpdsynonym']))
                for syn in synonyms:
                    name_cid_map[self.__normalize_chem_name(syn)] = cid
        
        return name_cid_map
    
    def __parse_reaction_str(self, reaction_str: str, name_cid_map: dict):
        reactions_str_split = reaction_str.split('->')
        if len(reactions_str_split) != 2:
            return None
        
        reagents, products = reactions_str_split

        parse_success = True

        unmapped_names = set()

        reagents_clean = []
        for chem_name in reagents.split('+'):
            norm_name = self.__normalize_chem_name(chem_name)
            cid = name_cid_map.get(norm_name)
            if cid is None:
                parse_success = False
                unmapped_names.add(norm_name)
            reagents_clean.append({'name': norm_name, 'cid': cid})
        
        products_clean = []
        for chem_name in products.split('+'):
            norm_name = self.__normalize_chem_name(chem_name)
            cid = name_cid_map.get(norm_name)
            if cid is None:
                parse_success = False
                unmapped_names.add(norm_name)
            products_clean.append({'name': norm_name, 'cid': cid})


        return parse_success, {'reagents': reagents_clean, 'products': products_clean}, unmapped_names


    def map_raw_reactions_chems_to_cids(self):
        name_cid_map = self.__extract_name_cid_map()
        self.log(f"Name-CID map size: {len(name_cid_map)}")

        unmapped_names = set()

        parsed = []
        with open(self.raw_reactions_verdict_fn) as f:
            for line in f:
                entry = json.loads(line)
                valid = entry['valid']
                reaction = entry['reaction']
                if not valid:
                    continue
                parse_res = self.__parse_reaction_str(reaction, name_cid_map)
                if parse_res is None:
                    continue
                parse_success, parsed_reaction, unmapped_names_curr = parse_res
                unmapped_names.update(unmapped_names_curr)
                if parse_success:
                    parsed.append(parsed_reaction)
        
        with open(self.reactions_parsed_fn, 'w') as f:
            for react in parsed:
                f.write(json.dumps(react) + '\n')
        
        with open(self.unmapped_names_fn, 'w') as f:
            for chem_name in unmapped_names:
                f.write(chem_name + '\n')
        
        self.log(f"Successfully parsed {len(parsed)} reactions; unmapped names size: {len(unmapped_names)}")

            
    def fetch_unmapped_names_from_pubchem(self):
        with open(self.unmapped_names_fn) as f:
            unmapped_names = f.read().strip().split('\n')
        
        blacklist = set()
        if os.path.exists(self.unmapped_names_blacklisted_fn):
            with open(self.unmapped_names_blacklisted_fn) as f:
                blacklist = set(f.read().strip().split('\n'))
        
        with open(self.chems_fn, 'a') as f_out, open(self.unmapped_names_blacklisted_fn, 'a') as f_out_black:
            for chem_name in unmapped_names:
                if chem_name in blacklist:
                    continue
                fetched_chems = pcp.get_compounds(chem_name, 'name')
                if not fetched_chems:
                    f_out_black.write(chem_name + '\n')
                    f_out_black.flush()
                    self.log(f"Failed to fetch pubchem data for unmapped name '{chem_name}'")
                    continue
                chem = fetched_chems[0]
                chem_pc_data = {
                    'cid': chem.cid,
                    'cmpdname': chem.iupac_name,
                    'cmpdsynonym': chem.synonyms,
                    'mf': chem.molecular_formula,
                    'mw': chem.molecular_weight,
                    'charge': chem.charge,
                    'smiles': chem.smiles,
                    'inchi': chem.inchi,
                    'inchikey': chem.inchikey,
                    'complexity': chem.complexity
                }
                if chem:
                    f_out.write(json.dumps(chem_pc_data) + '\n')
                    f_out.flush()
                    self.log(f"Fetched '{chem_pc_data['cmpdname']}' for unmapped name '{chem_name}'")
                



    def deduplicate_raw_reactions(self):
        with open(self.raw_reactions_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        raw_reacts = dict()
        models = dict()
        for entry in entries:
            cid = entry['cid']
            reactions = entry['reactions']
            model = entry['model']
            if cid not in raw_reacts:
                raw_reacts[cid] = []
            raw_reacts[cid] += reactions
            models[cid] = model
        
        with open(self.raw_reactions_fn, 'w') as f:
            for cid in raw_reacts:
                entry = {'cid': cid, 'reactions': raw_reacts[cid], 'model': models[cid]}
                f.write(json.dumps(entry) + '\n')
    

    def process_raw_reactions(self, reactions_fn):
        with open(reactions_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        for entry in entries:
            reactions = entry['reactions']
            for react in reactions:
                react = react.strip()
                reagents, products = react.split('->')
                reagents = reagents.strip().split('+')
                products = products.strip().split('+')
    

    def organize_chems_file(self):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        unique_chems = []
        unique_cids = set()
        for chem in chems:
            cid = chem['cid']
            if cid in unique_cids:
                continue
            if chem['cmpdname'] is None:
                if chem['cmpdsynonym']:
                    synonyms = list(filter(lambda x: not re.search(r'\d{3,}', x), chem['cmpdsynonym']))
                    if not synonyms:
                        continue
                    chem['cmpdname'] = synonyms[0].lower()
                else:
                    continue
            if chem['charge'] != 0:
                continue

            unique_cids.add(cid)
            unique_chems.append(chem)
        
        unique_chems.sort(key=lambda x: x['complexity'])

        with open(self.chems_fn, 'w') as f:
            for chem in unique_chems:
                f.write(json.dumps(chem) + '\n')



if __name__ == "__main__":
    chemsllm = ChemsLLM("data/")
    #chemsllm.get_raw_reactions(max_workers=20)
    #chemsllm.process_raw_reactions('raw_reactions.jsonl')
    #chemsllm.deduplicate_raw_reactions()
    #chemsllm.validate_raw_reactions(max_workers=20)
    #chemsllm.fix_broken_raw_reactions(max_workers=1)
    #print(chemsllm.find_all_unicode_chars_in_raw_reactions())
    chemsllm.map_raw_reactions_chems_to_cids()
    #chemsllm.fetch_unmapped_names_from_pubchem()
    #chemsllm.organize_chems_file()
    
