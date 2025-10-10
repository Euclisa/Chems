from openai import OpenAI
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import pubchempy as pcp
import shutil
from chempy import balance_stoichiometry
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, GraphDescriptors, rdMolDescriptors, inchi
import hashlib
import psycopg2
from psycopg2.extras import execute_values
import base64
import random
from time import sleep
import periodictable
import requests

# Disable all RDKit warnings and info messages
RDLogger.DisableLog('rdApp.*')


class ChemsLLM:
    def __init__(self, data_dir, db_name, api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.db_name = db_name

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        self.completion_tokens_total = 0
        self.input_tokens_total = 0
        self.tokens_total_lock = threading.Lock()
        
        self.print_lock = threading.Lock()

        self.gpt_oss = "openai/gpt-oss-120b"
        self.qwen = "qwen/qwen3-235b-a22b"
        self.grok = "x-ai/grok-3-mini"
        self.gemeni = "google/gemini-2.5-flash-lite"

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.structures_dir = os.path.join(self.data_dir, 'structures')
        if not os.path.exists(self.structures_dir):
            os.makedirs(self.structures_dir)

        self.raw_reactions_fn = os.path.join(self.data_dir, 'raw_reactions', "raw_reactions.jsonl")
        self.wiki_raw_reactions_fn = os.path.join(self.data_dir, 'raw_reactions', "wiki_raw_reactions.jsonl")
        self.top_rare_raw_reactions_fn = os.path.join(self.data_dir, 'raw_reactions', "top_rare_raw_reactions.jsonl")
        self.raw_reactions_verdict_fn = os.path.join(self.data_dir, 'raw_reactions', "raw_reactions_verdict.jsonl")
        self.raw_reactions_staged_fn = os.path.join(self.data_dir, 'raw_reactions', "raw_reactions_staged.jsonl")
        self.products_wiki_raw_reactions_fn = os.path.join(self.data_dir, 'raw_reactions', 'wiki_products_raw_reactions.jsonl')

        self.reactions_parsed_fn = os.path.join(self.data_dir, 'reactions_parsed', "reactions_parsed.jsonl")
        self.reactions_parsed_llm_fn = os.path.join(self.data_dir, 'reactions_parsed', "reactions_parsed_llm.jsonl")
        self.reactions_parsed_balanced_fn = os.path.join(self.data_dir, 'reactions_parsed', "reactions_parsed_balanced.jsonl")
        self.reactions_parsed_ord_fn = os.path.join(self.data_dir, 'reactions_parsed', 'reactions_parsed_ord.jsonl')

        self.chems_descriptions_fn = os.path.join(self.data_dir, 'chems', 'chems_descriptions.jsonl')
        self.chems_fn = os.path.join(self.data_dir, 'chems', "chems.jsonl")
        self.chems_categories_fn = os.path.join(self.data_dir, 'chems', "chems_categories.jsonl")
        self.wiki_chems_fn = os.path.join(self.data_dir, 'chems', "wiki_chems.jsonl")
        self.hazards_chems_fn = os.path.join(self.data_dir, 'chems', "hazards_chems.jsonl")
        self.chems_edges_fn = os.path.join(self.data_dir, 'chems', 'chems_edges.jsonl')
        self.elements_fn = os.path.join(self.data_dir, 'chems', 'elements.jsonl')

        self.categories_fn = os.path.join(self.data_dir, 'misc', "categories.jsonl")
        self.background_cids_fn = os.path.join(self.data_dir, 'misc', 'background_cids.json')
        self.commonness_sorted_cids_fn = os.path.join(self.data_dir, 'misc', 'commonness_sorted_cids.json')
        self.cids_blacklist_fn = os.path.join(self.data_dir, 'misc', 'cids_blacklist.jsonl')

        self.reactions_details_fn = os.path.join(self.data_dir, 'reactions_details', 'reactions_details.jsonl')
        self.reactions_details_ord_fn = os.path.join(self.data_dir, 'reactions_details', 'reactions_details_ord.jsonl')
        self.reactions_details_llm_fn = os.path.join(self.data_dir, 'reactions_details', 'reactions_details_llm.jsonl')
        self.reactions_descriptions_fn = os.path.join(self.data_dir, 'reactions_details', 'reactions_descriptions.jsonl')

        self.unmapped_names_fn = os.path.join(self.data_dir, "unmapped_names.txt")
        self.chem_names_blacklisted_fn = os.path.join(self.data_dir, "unmapped_names_blacklisted.txt")
        self.unbalancing_cids_fn = os.path.join(self.data_dir, "unbalancing_cids.txt")
        self.unmapped_smiles_fn = os.path.join(self.data_dir, 'unmapped_smiles.txt')
        self.chem_smiles_blacklisted_fn = os.path.join(self.data_dir, 'unmapped_smiles_blacklisted.txt')

        self.cids_blacklist = set([x['cid'] for x in self.__load_jsonl(self.cids_blacklist_fn)])

        self.__chems = None
        self.__cid_chem_map = None
        self.__inchikey_chem_map = None
        self.__name_cid_map = None
        self.__cid_mf_map = None

        self.unmapped_names_delimiter = "||"

        self.complexity_thr = 550
        self.max_synonyms_thr = 100
        
        self.sources_priority = {
            "ord": 10,
            self.gpt_oss: 5,
            self.qwen: 4,
            self.grok: 3
        }


    def __update_chems(self, new_chems=None):
        if new_chems is not None:
            self.__write_jsonl(new_chems, self.chems_fn)

        self.__chems = new_chems
        self.__cid_chem_map = None
        self.__inchikey_chem_map = None
        self.__name_cid_map = None
        self.__cid_mf_map = None

    @property
    def chems(self):
        if self.__chems is None:
            self.__chems = self.__load_jsonl(self.chems_fn)
        
        return self.__chems

    @property
    def cid_chem_map(self):
        if self.__cid_chem_map is None:
            self.__cid_chem_map = {chem['cid']: chem for chem in self.chems}
        
        return self.__cid_chem_map
    
    @property
    def inchikey_chem_map(self):
        if self.__inchikey_chem_map is None:
            self.__inchikey_chem_map = {chem['inchikey']: chem for chem in self.chems}
        
        return self.__inchikey_chem_map
    
    @property
    def name_cid_map(self):
        if self.__name_cid_map is None:
            self.__name_cid_map = dict()
            with open(self.chems_fn) as f:
                for line in f:
                    entry = json.loads(line)
                    cid = entry['cid']
                    self.__name_cid_map[self.__normalize_chem_name(entry['cmpdname'], is_clean=True)] = cid
                    synonyms = list(filter(lambda x: not re.search(r'\d{3,}', x), entry['cmpdsynonym']))
                    for syn in synonyms:
                        self.__name_cid_map[self.__normalize_chem_name(syn, is_clean=True)] = cid
        
        return self.__name_cid_map
    

    @property
    def cid_mf_map(self):
        if self.__cid_mf_map is None:
            self.__cid_mf_map = {chem['cid']: chem['mf'] for chem in self.chems}
        
        return self.__cid_mf_map


    def log(self, message=""):
        with self.print_lock:
            print(message)
    
    def __fetch_llm_answer(self, messages, model, reasoning_effort="medium"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort
            )

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
    
    
    def __fetch_raw_reactions(self, chem, mode="documented_rp"):
        try:
            chem_name = chem['cmpdname']
            
            if mode == "documented_rp":
                instruct = \
                f"Please, provide a comprehensive list of documented chemical reactions involving {chem_name}, where it appears as either a reactant or a product. " \
                "Write the reactions as schemes using only '->' and '+' symbols. Use the full chemical names of substances instead of formulas or generic terms. " \
                "Do not include balancing coefficients, comments, or any markup - only the reaction schemes themselves. " \
                "If no such substance exists or no documented reactions are available, return 'None'."
            elif mode == "documented_less_common_rp":
                instruct = \
                f"Please provide a comprehensive and diverse list of documented chemical reactions involving {chem_name}, where it appears as either a reactant or a product. " \
                "Include not only the most common reactions, but also less common or more unusual or exotic ones, as long as you are absolutely sure they are real and correct. " \
                "Write the reactions as schemes using only '->' and '+' symbols. Use the full chemical names of substances instead of formulas or generic terms. " \
                "Do not include balancing coefficients, comments, or any markup – only the reaction schemes themselves. " \
                "If no such substance exists or no documented reactions are available, return 'None'."
            elif mode == "documented_less_common_p":
                instruct = \
                f"Please provide a comprehensive and diverse list of documented chemical reactions involving {chem_name}, where it appears as a product. " \
                "Include not only the most common reactions, but also less common or more unusual or exotic ones, as long as you are absolutely sure they are real and correct. " \
                "Write the reactions as schemes using only '->' and '+' symbols. Use the full chemical names of substances instead of formulas or generic terms. " \
                "Do not include balancing coefficients, comments, or any markup – only the reaction schemes themselves. " \
                "If no such substance exists or no documented reactions are available, return 'None'."
            elif mode == "rare_rp":
                instruct = \
                f"Please provide a comprehensive and diverse list of chemical reactions involving {chem_name}, where it appears as either a reactant or a product. " \
                "Include only the uncommon, rare and exotic reactions, but you must be absolutely sure they are real and correct. " \
                "Write the reactions as schemes using only '->' and '+' symbols. Use the full chemical names of substances instead of formulas or generic terms. " \
                "Do not include balancing coefficients, comments, or any markup – only the reaction schemes themselves. " \
                "If no such substance exists or no documented reactions are available, return 'None'."
            else:
                raise Exception(f"Invalid raw reactions generation mode: {mode}")

            instruct_revalidate = \
            "Please, review the provided reactions. Identify any erroneous reactions and correct them where possible. Return revised reactions list that comply with the initial requirements."


            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instruct}
            ]

            models_schedule = [self.gpt_oss, self.qwen]
            
            for curr_model in models_schedule:
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
            
            self.log(f"Got {len(reactions_revalid)} reactions for {chem_name} with '{model}'; CTT: {self.completion_tokens_total}")
        
        except Exception as e:
            self.log(f"Exception in '__fetch_raw_reactions': {e}")
            return None
        
        return {'cid': chem['cid'], 'reactions': reactions_revalid}



    def get_raw_reactions(self, max_workers=1):        
        processed = self.__get_processed_entries(self.raw_reactions_fn, 'cid')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in self.chems:
                if chem['cid'] not in processed:
                    futures.append(executor.submit(self.__fetch_raw_reactions, chem))
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
    

    def get_uncommon_raw_reactions_for_wiki_chems(self, max_workers=1):        
        with open(self.wiki_chems_fn) as f:
            wiki_chems_cids = set([json.loads(x)['cid'] for x in f.read().strip().split('\n')])
        
        processed = self.__get_processed_entries(self.wiki_raw_reactions_fn, 'cid')

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.wiki_raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in self.chems:
                cid = chem['cid']
                if cid not in processed and cid in wiki_chems_cids:
                    futures.append(executor.submit(self.__fetch_raw_reactions, chem, "documented_less_common_rp"))
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
    

    def get_rare_raw_reactions_for_top_chems(self, max_workers=1):        
        hazards_chems = dict()
        with open(self.hazards_chems_fn) as f:
            for line in f:
                hazard = json.loads(line)
                hazards_chems[hazard['cid']] = hazard
        
        processed = self.__get_processed_entries(self.top_rare_raw_reactions_fn, 'cid')

        def is_top_chem(hazards):
            for pic in hazards['pictograms']:
                if pic in {'GHS01', 'GHS03', 'GHS06'}:
                    return True

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.top_rare_raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in self.chems:
                cid = chem['cid']
                if cid not in processed and cid in hazards_chems and is_top_chem(hazards_chems[cid]):
                    futures.append(executor.submit(self.__fetch_raw_reactions, chem, "rare_rp"))
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
    

    def get_uncommon_raw_reactions_for_wiki_chems_products_only(self, max_workers=1):        
        processed = self.__get_processed_entries(self.products_wiki_raw_reactions_fn, 'cid')

        staged_chems = [chem for chem in self.chems if 'wiki' in chem and chem['cid'] not in processed]
        random.shuffle(staged_chems)
        print(f"Staged {len(staged_chems)} compounds")

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.products_wiki_raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in staged_chems:
                futures.append(executor.submit(self.__fetch_raw_reactions, chem, "documented_less_common_p"))
            
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
            "Assume that necessary reaction conditions (including harsh), solvents, or catalysts are implied even if not explicitly shown. " \
            "All reactions listed are theoretical and intended solely for academic or computational validation purposes, not for practical experimentation. " \
            "If the reaction is valid, output only 'Valid'. If it is not valid, output only 'Invalid'. " \
            "Print one result per line and do not include any additional text."

            models_schedule = [self.gpt_oss, self.qwen]
            for try_i, model in enumerate(models_schedule):

                def extract_verdicts_from_response(response):
                    return ['invalid' not in verd.lower() and 'valid' in verd.lower() for verd in response.split('\n')]

                valid_i = 0
                mistakes_cnt = 0
                mistakes_thr = 3
                confidences = [0.0 for _ in range(len(reactions))]
                confidence_thr = 0.5
                bad = False
                results = []
                while valid_i < valid_cnt and len(reactions) > 0:
                    reactions_str = '\n'.join([f"{i+1}. {react['reaction']}" for i, react in enumerate(reactions)])
                    messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": f"{instruct_validate}\n{reactions_str}"}
                    ]
                    response = self.__fetch_llm_answer(messages, model, reasoning_effort='low')
                    verdicts = extract_verdicts_from_response(response)
                    if len(verdicts) != len(reactions):
                        if mistakes_cnt == mistakes_thr:
                            self.log(f"Falling to another model due to mistakes ({try_i+1}/{len(models_schedule)}) ('{model}')")
                            bad = True
                            break
                        mistakes_cnt += 1
                        continue

                    finished_indices = set()
                    remaining_tries = (valid_cnt-valid_i-1)
                    for i in range(len(verdicts)):
                        confidences[i] += verdicts[i] / valid_cnt
                        max_confidence = remaining_tries / valid_cnt + confidences[i]
                        est_confidence = (confidences[i] + max_confidence) / 2
                        if confidences[i] >= confidence_thr or max_confidence <= confidence_thr:
                            react = reactions[i].copy()
                            reaction_str = react['reaction']
                            react['valid'] = confidences[i] > confidence_thr
                            react['confidence'] = confidences[i]
                            react['source'] = model
                            results.append(react)
                            finished_indices.add(i)
                            self.log(f"Processed reaction '{reaction_str}'; confidence: {est_confidence:.2f}; CTT: {self.completion_tokens_total}")

                    reactions = [reactions[i] for i in range(len(reactions)) if i not in finished_indices]
                    confidences = [confidences[i] for i in range(len(confidences)) if i not in finished_indices]
                    
                    valid_i += 1

                if bad:
                    continue
                
                return results
        
        except Exception as e:
            self.log(f"Exception in '__validate_raw_reaction': {e}")
            return None
    

    def validate_raw_reactions(self, raw_reactions_fn=None, max_workers=1):
        if raw_reactions_fn is None:
            raw_reactions_fn = self.raw_reactions_fn

        entries = self.__load_jsonl(raw_reactions_fn)
        
        processed = self.__get_processed_entries(self.raw_reactions_verdict_fn, 'reaction')
        
        reactions = []

        for entry in entries:
            cid = entry['cid']
            reactions_curr = entry['reactions']
            for react in reactions_curr:
                if react not in processed:
                    reactions.append({'cid': cid, 'reaction': react})
        
        reactions_batch_size = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.raw_reactions_verdict_fn, 'a') as f_out:
            futures = []
            i = 0
            while i < len(reactions):
                futures.append(executor.submit(self.__validate_raw_reactions, reactions[i:i+reactions_batch_size], 9))
                i += reactions_batch_size
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    for react in res:
                        f_out.write(json.dumps(react) + '\n')
                    f_out.flush()
    

    def __extract_good_synonyms(self, chem, max_syns):
        syns = chem['cmpdsynonym']
        good_syns = []
        for syn in syns:
            if re.search(r'\d{3,}', syn):
                continue
            if re.search(r'SCHEMBL', syn):
                continue
            good_syns.append(syn)

        return good_syns[:max_syns]
    

    def __get_chem_description(self, chem, valid_cnt):
        try:
            chem_name = chem['cmpdname']
    
            instruct = \
            "Write an engaging and informative plain text description of the compound these synonyms refer to. " \
            "You may decide freely what aspects to include — composition, properties, uses, history, relevance, or anything else meaningful — " \
            "as long as the result is interesting to read and based on reliable chemical knowledge. " \
            "If nothing meaningful can be said about it, respond exactly with the word None.\n" \
            "Result should be in plain text format. Markdown or other formatting types are strictly forbidden." \
            "Guidelines:\n" \
            "- Everything you write must be true or based on well-accepted chemical knowledge.\n" \
            "- You may include lesser-known but reliable information, but do not guess or invent facts.\n" \
            "- The text should feel lively, colorful, and pleasant to read — avoid dull or purely technical writing.\n" \
            "- You may write at any length, as long as the text remains engaging and free of boring or redundant details.\n"

            instruct_validate = \
            "Decide if the following text is factually correct and does not contain misleading or speculative information. " \
            "If the description is fully correct or at least safely accurate within general chemical knowledge, return 'Valid'. " \
            "If it contains anything blatantly wrong, incorrect, or unverified, return 'Invalid'. " \
            "Do not write anything except 'Valid' or 'Invalid'."

            instruct_fix = \
            "Review the provided chemical compound description for errors or inconsistencies. " \
            "Identify any mistakes and return a corrected, accurate version of the description, preserving its original narrative tone. " \
            "Return only the corrected description, without any explanations or extra text."

            av_confidence = 0
            synonyms = ', '.join(list(map(lambda x: f'"{x}"', self.__extract_good_synonyms(chem, 3))))
            models_schedule = [self.gpt_oss, self.gpt_oss, self.qwen, self.qwen]
            for try_i, model in enumerate(models_schedule):
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{synonyms}\n\n{instruct}"}
                ]

                description = self.__fetch_llm_answer(messages, model)
                
                if len(description) < 20:
                    self.log(f"Failed to generate description for {chem_name} on try {try_i+1}/{len(models_schedule)} ('{model}')")
                    continue
                

                def validate_description(descr, confidence_thr):
                    nonlocal instruct_validate, model

                    messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": f"{instruct_validate}\n\n{descr}"}
                    ]
                    confidence = 0
                    for valid_i in range(valid_cnt):
                        verdict = self.__fetch_llm_answer(messages, model)
                        if 'invalid' not in verdict.lower():
                            if 'valid' in verdict.lower():
                                confidence += 1 / valid_cnt

                        remaining_tries = (valid_cnt-valid_i-1)
                        max_confidence = remaining_tries / valid_cnt + confidence
                        if max_confidence < confidence_thr or confidence >= confidence_thr:
                            confidence += remaining_tries / 2 / valid_cnt
                            break

                    return confidence
                
                confidence_thr = 0.49
                confidence = validate_description(description, confidence_thr)
                
                if confidence >= confidence_thr:
                    self.log(f"('{model}') Generated description for {chem_name} of length {len(description)}. confidence: {confidence}; CTT: {self.completion_tokens_total}")
                    return {'cid': chem['cid'], 'description': description, 'confidence': confidence, 'source': model}

                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{instruct_fix}\n\n{description}"}
                ]
                description = self.__fetch_llm_answer(messages, model)
                confidence = validate_description(description, confidence_thr)
                if confidence >= confidence_thr:
                    self.log(f"('{model}') Generated description for {chem_name} of length {len(description)} after fixing. confidence: {confidence}; CTT: {self.completion_tokens_total}")
                    return {'cid': chem['cid'], 'description': description, 'confidence': confidence, 'source': model}

                self.log(f"Low validation confidence for {chem_name} on try {try_i+1}/{len(models_schedule)} ('{model}'): {confidence}")
                av_confidence += confidence / len(models_schedule)
            
            self.log(f"Failed to generate description for {chem_name} due to low validation confidence: {av_confidence}")
                
        except Exception as e:
            self.log(f"Exception during description generation for '{chem_name}': {e}")

        return None
        
    

    def get_chems_descriptions(self, max_workers=1):
        chems_power = dict()
        for chem in self.chems:
            chems_power[chem['cid']] = 0

        with open(self.reactions_parsed_balanced_fn) as f:
            for line in f:
                reaction = json.loads(line)
                all_cids = [x['cid'] for x in reaction['reagents']] + [x['cid'] for x in reaction['products']]
                for cid in all_cids:
                    chems_power[cid] += 1
        
        with open(self.hazards_chems_fn) as f:
            hazard_cids = set([json.loads(x)['cid'] for x in f.read().strip().split('\n')])
        
        self.chems.sort(key=lambda x: chems_power[x['cid']], reverse=True)

        processed = self.__get_processed_entries(self.chems_descriptions_fn, 'cid')

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.chems_descriptions_fn, 'a') as f_out:
            futures = []
            for chem in self.chems:
                if chem['cid'] not in processed and 'wiki' in chem:
                    futures.append(executor.submit(self.__get_chem_description, chem, 6))
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    f_out.write(json.dumps(res) + '\n')
                    f_out.flush()
    

    def __get_reactions_description(self, reactions, valid_cnt):
        try:
            instruct = \
            "Please generate detailed short plain text descriptions for each of the chemical reactions provided below. " \
            "For each reaction, include the general type of reaction, its purpose if applicable, key transformations, typical reaction conditions, " \
            "and common solvents or catalysts if needed. Prefix each description with the corresponding reaction number. " \
            "Provide only the descriptions, do not add any extra text or commentary and do not restate or repeat the reactions themselves. Provide only information that you are confident is correct. " \
            "Do not use formatting, but plain text only. If you cannot provide reliable information for a specific reaction, simply write 'None' as the description."
            
            instruct_validate = \
            "You will be given pairs of chemical reaction schemes and their descriptions. Each reaction scheme shows reagents and products (unbalanced). " \
            "Your task is to validate whether each description accurately describes the corresponding reaction. " \
            "Return 'Valid' if the description is chemically accurate and contains no significant errors. " \
            "Return 'Invalid' if the description contains factual errors, incorrect mechanisms, impossible conditions, or misidentified reaction types. " \
            "Minor imprecision is acceptable if the core chemistry is correct. " \
            "Respond with only 'Valid' or 'Invalid' for each pair, one per line, in the same order as presented."

            instruct_fix = \
            "You will be given pairs of chemical reaction schemes and their descriptions. " \
            "Find and correct the chemical errors if any in each description while keeping all other text as similar as possible to the original. " \
            "Provide only the corrected descriptions, one per line, in the same order as presented. " \
            "If description is wrong and you cannot reliably correct a description, write 'None'."

            models_schedule = [self.gpt_oss, self.qwen]
            for try_i, model in enumerate(models_schedule):
                reactions_formatted_str = '\n'.join([f"{i}. {react['reaction']}" for i, react in enumerate(reactions)])
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{instruct}\n\n{reactions_formatted_str}"}
                ]

                def extract_descriptions_from_response(response):
                    descriptions = re.split(r'\n(?=\d+[.)]\s)', response)
                    descriptions = [d.strip() for d in descriptions if d.strip()]
                    descriptions = [re.sub(r'^\d+[.)]\s+', '', d) for d in descriptions]
                    return descriptions
                
                response = self.__fetch_llm_answer(messages, model)
                descriptions = extract_descriptions_from_response(response)
                if len(descriptions) != len(reactions):
                    self.log(f"Failed to get descriptions ({len(descriptions)} != {len(reactions)}) ('{model}') ({try_i+1}/{len(models_schedule)})")
                    continue

                def filter_descriptions_reactions(descriptions, reactions):
                    indices = [i for i, desc in enumerate(descriptions) if len(desc) > 20]
                    return [descriptions[i] for i in indices], [reactions[i] for i in indices]

                descriptions, reactions = filter_descriptions_reactions(descriptions, reactions)
                
                def extract_verdicts_from_response(response):
                    return ['invalid' not in verd.lower() and 'valid' in verd.lower() for verd in response.split('\n')]

                valid_i = 0
                mistakes_cnt = 0
                mistakes_thr = 2
                confidences = [0 for _ in range(len(descriptions))]
                confidence_thr = 0.39
                results = []
                while valid_i < valid_cnt and len(descriptions) > 0:
                    formatted_descriptions_str = []
                    for i in range(len(descriptions)):
                        react = reactions[i]['reaction']
                        formatted_descriptions_str.append(f'{i+1}. Scheme: "{react}"\nDescription: "{descriptions[i]}"')
                    formatted_descriptions_str = '\n\n'.join(formatted_descriptions_str)
                    messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": f"{instruct_validate}\n\n{formatted_descriptions_str}"}
                    ]
                    response = self.__fetch_llm_answer(messages, model)
                    verdicts = extract_verdicts_from_response(response)
                    if len(verdicts) != len(descriptions):
                        if mistakes_cnt == mistakes_thr:
                            self.log("Returning early due to mistakes at validation phase")
                            return results
                        mistakes_cnt += 1
                        continue
                    
                    finished_indices = set()
                    remaining_tries = (valid_cnt-valid_i-1)
                    for i in range(len(verdicts)):
                        confidences[i] += verdicts[i] / valid_cnt
                        max_confidence = remaining_tries / valid_cnt + confidences[i]
                        est_confidence = (confidences[i] + max_confidence) / 2
                        if confidences[i] >= confidence_thr:
                            reaction_str = reactions[i]['reaction']
                            self.log(f"Generated description for '{reaction_str}'; confidence: {est_confidence}")
                            results.append({'rid': reactions[i]['rid'], 'description': descriptions[i], 'confidence': est_confidence, 'source': model})
                            finished_indices.add(i)
                        elif max_confidence < confidence_thr:
                            finished_indices.add(i)
                    reactions = [reactions[i] for i in range(len(reactions)) if i not in finished_indices]
                    descriptions = [descriptions[i] for i in range(len(descriptions)) if i not in finished_indices]
                    confidences = [confidences[i] for i in range(len(confidences)) if i not in finished_indices]

                    valid_i += 1
                
                return results

        
        except Exception as e:
            self.log(f"Exception during reactions description generation: {e}")
        
        return None
    

    def __get_reaction_as_str(self, reaction):
        def format_components(components):
            parts = []
            for c in components:
                coeff = f"{c['coeff']} " if c.get('coeff') is not None else ""
                parts.append(f"{coeff}{c['original_name']}")
            return " + ".join(parts)

        reagents_str = format_components(reaction['reagents'])
        products_str = format_components(reaction['products'])

        return f"{reagents_str} -> {products_str}"

    def get_reactions_descriptions(self, max_workers=1):
        chems_power = dict()
        for chem in self.chems:
            chems_power[chem['cid']] = 0

        reactions = []
        with open(self.reactions_parsed_balanced_fn) as f:
            for line in f:
                reaction = json.loads(line)
                all_cids = [x['cid'] for x in reaction['reagents']] + [x['cid'] for x in reaction['products']]
                for cid in all_cids:
                    chems_power[cid] += 1
                reactions.append(reaction)

        reactions_power = dict()
        for react in reactions:
            all_cids = [x['cid'] for x in reaction['reagents']] + [x['cid'] for x in reaction['products']]
            reactions_power[react['rid']] = sum(chems_power[x] for x in all_cids) / len(all_cids)
        
        processed = self.__get_processed_entries(self.reactions_descriptions_fn, 'rid')
        reactions = list(filter(lambda x: x['rid'] not in processed and x['source'] != 'ord', reactions))
        reactions.sort(key=lambda x: reactions_power[x['rid']], reverse=True)

        reactions_batch_size = 6
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.reactions_descriptions_fn, 'a') as f_out:
            futures = []
            for i in range(0, len(reactions), reactions_batch_size):
                reactions_arg = [{'reaction': self.__get_reaction_as_str(x), 'rid': x['rid']} for x in reactions[i:i+reactions_batch_size]]
                futures.append(executor.submit(self.__get_reactions_description, reactions_arg, 6))
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    self.log(f"\nLost {reactions_batch_size-len(res)}/{reactions_batch_size}; CTT: {self.completion_tokens_total}\n")
                    for entry in res:
                        f_out.write(json.dumps(entry) + '\n')
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
    

    def __chem_name_to_ascii(self, chem_name_raw):
        unicode_map = {
            '‑': '-',
            'α': 'alpha',
            'γ': 'gamma,',
            '–': '-',
            '\u2019': "'"
        }
        chem_name_ascii = ""
        for char in chem_name_raw:
            if not char.isascii():
                if char in unicode_map:
                    char = unicode_map[char]
                else:
                    char = ""
            chem_name_ascii += char
        
        return chem_name_ascii
    

    def __clean_chem_name(self, chem_name_raw, is_clean=False):
        chem_name = self.__chem_name_to_ascii(chem_name_raw)
        chem_name = chem_name.strip()
        chem_name = re.sub(r'\s+', ' ', chem_name)

        if not is_clean:
            chem_name = chem_name.strip('`\'".,;:')
            chem_name = re.sub(r'^\d+ ', '', chem_name)

        return chem_name


    def __normalize_chem_name(self, chem_name_raw, is_clean=False):
        chem_name = self.__clean_chem_name(chem_name_raw, is_clean=is_clean)
        chem_name = chem_name.lower()
        chem_name = chem_name.strip()
        chem_name = chem_name.replace("aluminum", "aluminium")
        
        if not is_clean:
            chem_name = re.sub(r' \([^\d]+\)$', '', chem_name)
            chem_name = chem_name.replace(' vapor', '')
            chem_name = chem_name.replace(' dust', '')
            chem_name = chem_name.replace('solution', '')
            chem_name = chem_name.replace('concentrated', '')
            chem_name = chem_name.replace('dilute ', '')
            chem_name = chem_name.replace('fuming ', '')
            chem_name = chem_name.replace('solid', '')
            chem_name = chem_name.replace('glacial ', '')
            chem_name = chem_name.replace('elemental', '')
            chem_name = chem_name.replace(' metal', '')
            chem_name = chem_name.replace('aqueous', '')
            chem_name = chem_name.replace(' gas', '')
            chem_name = chem_name.replace('hot ', '')
            chem_name = chem_name.replace('uv light', 'light')
            chem_name = chem_name.replace('blue light', 'light')
            chem_name = chem_name.replace('ultraviolet light', 'light')
            
            if "catalyst" in chem_name or 'raney nickel' in chem_name:
                chem_name = "catalyst"

        chem_name = re.sub(r'\s+', '', chem_name)

        return chem_name

    
    def __get_reaction_hash(self, reaction):
        reagents_cids = sorted([x['cid'] for x in reaction['reagents']])
        products_cids = sorted([x['cid'] for x in reaction['products']])
        reagents_str = '(' + ','.join([str(x) for x in reagents_cids]) + ')'
        products_str = '(' + ','.join([str(x) for x in products_cids]) + ')'
        reaction_enc = (reagents_str + products_str).encode("utf-8")
        hash_bytes = hashlib.sha256(reaction_enc).digest()
        hash_b64 = base64.b64encode(hash_bytes[:16]).decode("utf-8")

        return hash_b64
    

    def __get_reaction_complexity(self, reaction):
        av_complexity = 0
        for chem in reaction['products']:
            av_complexity += self.cid_chem_map[chem['cid']]['complexity']
        
        for chem in reaction['reagents']:
            av_complexity += self.cid_chem_map[chem['cid']]['complexity']
        av_complexity /= len(reaction['reagents']) + len(reaction['products'])

        return av_complexity


    def __assemble_reaction(self, reaction):
        reaction['complexity'] = self.__get_reaction_complexity(reaction)
        reaction['rid'] = self.__get_reaction_hash(reaction)

        return reaction


    
    def __parse_reaction_str(self, reaction_str: str):
        parts = reaction_str.split('->')
        if len(parts) != 2:
            return None, set()

        reagents_str, products_str = parts
        parse_success = True
        unmapped_names = set()

        def parse_compounds(compound_str, skip_names, existing_cids=None):
            if existing_cids is None:
                existing_cids = set()

            compounds, cids = [], set()
            for name in compound_str.split('+'):
                norm = self.__normalize_chem_name(name)
                if norm in skip_names:
                    continue

                clean = self.__clean_chem_name(name)
                cid = self.name_cid_map.get(norm)
                if cid is None:
                    nonlocal parse_success
                    parse_success = False
                    unmapped_names.add((norm, clean))

                if cid is None or cid not in existing_cids | cids:
                    compounds.append({'norm_name': norm, 'original_name': clean, 'cid': cid})
                    cids.add(cid)

            return list({c["cid"]: c for c in compounds}.values()), cids

        reagents, reagents_cids = parse_compounds(reagents_str, {"light", "heat", "catalyst"})
        products, products_cids = parse_compounds(products_str, {"otherproducts"}, reagents_cids)

        if products_cids & reagents_cids or not products or not reagents:
            parse_success = False

        if not parse_success:
            return None, unmapped_names

        reaction = {'reagents': reagents, 'products': products}
        reaction = self.__assemble_reaction(reaction)

        return reaction, unmapped_names


    def map_raw_reactions_chems_to_cids(self):
        unmapped_names = dict()
        parsed = []
        processed_rids = set()
        with open(self.raw_reactions_verdict_fn) as f:
            for line in f:
                entry = json.loads(line)
                reaction = entry['reaction']
                confidence = entry['confidence']
                if confidence < 0.4:
                    continue
                parsed_reaction, unmapped_names_curr = self.__parse_reaction_str(reaction)
                for norm_name, name in unmapped_names_curr:
                    if norm_name not in unmapped_names:
                        unmapped_names[norm_name] = [0, name]
                    unmapped_names[norm_name][0] += 1
                if not parsed_reaction:
                    continue
                
                processed_rids.add(parsed_reaction['rid'])
                
                parsed_reaction['confidence'] = confidence
                parsed_reaction['source'] = entry['source']
                parsed.append(parsed_reaction)
        
        self.__write_jsonl(parsed, self.reactions_parsed_llm_fn)
        
        unmapped_names_list = list(unmapped_names.keys())
        unmapped_names_list.sort(key=lambda x: unmapped_names[x][0], reverse=True)
        with open(self.unmapped_names_fn, 'w') as f:
            for chem_name in unmapped_names_list:
                f.write(f"{chem_name}{self.unmapped_names_delimiter}{unmapped_names[chem_name][1]}{self.unmapped_names_delimiter}{unmapped_names[chem_name][0]}\n")
        
        self.log(f"Successfully parsed {len(parsed)} reactions; unmapped names size: {len(unmapped_names)}")

            
    def fetch_names_from_pubchem(self, names_fn):
        with open(names_fn) as f:
            names_list = f.read().strip().split('\n')
        
        blacklist = set()
        if os.path.exists(self.chem_names_blacklisted_fn):
            with open(self.chem_names_blacklisted_fn) as f:
                blacklist = set(map(lambda x: self.__normalize_chem_name(x), f.read().strip().split('\n')))
        
        with open(self.chems_fn, 'a') as f_out, open(self.chem_names_blacklisted_fn, 'a') as f_out_black:
            for entry in names_list:
                chem_name_norm, chem_name, cnt = entry.split(self.unmapped_names_delimiter)
                if chem_name_norm in self.name_cid_map:
                    continue
                if chem_name_norm in blacklist:
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

                if chem.inchikey in self.inchikey_chem_map:
                    curr_cid = self.inchikey_chem_map[chem.inchikey]['cid']
                    self.log(f"Compound with name '{chem_name}' has equivalent compound in chem file (CID: {curr_cid})")
                    continue

                f_out.write(json.dumps(chem_pc_data) + '\n')
                f_out.flush()
                self.log(f"Fetched '{chem_pc_data['cmpdname']}' for name '{chem_name}'")
        
    
    def fetch_chems_cids_from_pubchem(self, cids):
        with open(self.chems_fn, 'a') as f_out:
            for cid in cids:
                if cid in self.cid_chem_map:
                    self.log(f"CID {cid} is in the black list")
                    continue

                fetched_chems = pcp.get_compounds(cid, 'cid')
                if not fetched_chems:
                    self.log(f"Failed to fetch pubchem data for cid '{cid}'")
                    continue
                
                chem = fetched_chems[0]

                if chem.inchikey in self.inchikey_chem_map:
                    curr_cid = self.inchikey_chem_map[chem.inchikey]['cid']
                    self.log(f"Compound with CID {cid} has equivalent compound in chem file (CID: {curr_cid})")
                    continue

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

                f_out.write(json.dumps(chem_pc_data) + '\n')
                f_out.flush()
                self.log(f"Fetched '{chem_pc_data['cmpdname']}' for CID '{cid}'")



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
    

    def organize_chems_file(self, force=False):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")
        
        unique_inchikeys_chems = dict()
        for chem in self.chems:
            cid = chem['cid']
            if chem['charge'] != 0 or cid in self.cids_blacklist or chem['complexity'] > self.complexity_thr:
                continue
            
            try:
                mol = Chem.MolFromSmiles(chem['smiles'])
                chem['smiles'] = Chem.MolToSmiles(mol, canonical=True)
            except:
                continue
            
            if chem['cmpdsynonym']:
                synonyms = list(filter(lambda x: not re.search(r'\d{3,}', x), chem['cmpdsynonym']))
                if not synonyms:
                    continue
                if chem['cmpdname'] is None:
                    chem['cmpdname'] = synonyms[0].lower()
                chem['cmpdsynonym'] = synonyms[:self.max_synonyms_thr]
            else:
                continue

            if 'ECFP4_fp' not in chem or force:
                chem['ECFP4_fp'] = self.__get_mol_fingerprint(mol)
            
            if 'bertz_complexity' not in chem or force:
                chem['bertz_complexity'] = self.__get_mol_bertz_complexity(mol)
            
            if 'organic' not in chem or force:
                chem['organic'] = self.__get_mol_organic_mark(mol)


            if 'inchi_snone' not in chem or force:
                chem['inchi_snone'] = inchi.MolToInchi(mol, options="/SNon")
            
            if 'inchikey_snone' not in chem or force:
                chem['inchikey_snone'] = inchi.MolToInchiKey(mol, options="/SNon")

            inchikey = chem['inchikey_snone']
            if inchikey in unique_inchikeys_chems:
                old_chem = unique_inchikeys_chems[inchikey]
                old_inchi = old_chem['inchi']
                curr_inchi = chem['inchi']
                if len(curr_inchi) < len(old_inchi):
                    unique_inchikeys_chems[inchikey] = chem
            else:
                unique_inchikeys_chems[inchikey] = chem
            
        
        unique_chems = sorted(list(unique_inchikeys_chems.values()), key=lambda x: x['complexity'])

        self.__update_chems(unique_chems)


    def balance_parsed_reactions(self, reactions_parsed_fn=None):
        if reactions_parsed_fn is None:
            reactions_parsed_fn = self.reactions_parsed_fn
            
        with open(reactions_parsed_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed_reactions = []
        balanced_cnt = 0
        for react in reactions:
            check_rid = self.__get_reaction_hash(react)
            if react['rid'] != check_rid:
                raise Exception(f"Bad RID for reaction '{react['rid']}'")

            reagents = [self.cid_mf_map[x['cid']] for x in react['reagents']]
            products = [self.cid_mf_map[x['cid']] for x in react['products']]
    
            try:
                reagents_coeffs, products_coeffs = balance_stoichiometry(reagents, products, underdetermined=False)
            except:
                react['balanced'] = False
                processed_reactions.append(react)
                continue

            all_coeffs = list(reagents_coeffs.values()) + list(products_coeffs.values())
            all_coeffs = [int(x) for x in all_coeffs]
            max_coeff = max(all_coeffs)
            if max_coeff > 30:
                react['balanced'] = False
                processed_reactions.append(react)
                continue

            for chem in react['reagents']:
                mf = self.cid_mf_map[chem['cid']]
                chem['coeff'] = int(reagents_coeffs[mf])
            
            for chem in react['products']:
                mf = self.cid_mf_map[chem['cid']]
                chem['coeff'] = int(products_coeffs[mf])
            
            react['balanced'] = True
            processed_reactions.append(react)
            balanced_cnt += 1
        
        with open(self.reactions_parsed_balanced_fn, 'w') as f:
            for react in processed_reactions:
                f.write(json.dumps(react) + '\n')
        
        self.log(f"Balanced {balanced_cnt} out of {len(reactions)}")
    

    def find_unbalancing_chems(self):
        unbalanced_cids = set()
        unbalanced_cids_cnt = dict()
        balanced_cids = set()
        cid_to_name = dict()
        with open(self.reactions_parsed_balanced_fn) as f:
            for line in f:
                entry = json.loads(line)
                cids = set()

                for chem in entry['reagents']:
                    cid = chem['cid']
                    name = chem['original_name']
                    cid_to_name[cid] = name
                    cids.add(cid)
                
                for chem in entry['products']:
                    cid = chem['cid']
                    name = chem['original_name']
                    cid_to_name[cid] = name
                    cids.add(cid)

                if entry['balanced']:
                    balanced_cids.update(cids)
                else:
                    unbalanced_cids.update(cids)
                    for cid in cids:
                        if cid not in unbalanced_cids_cnt:
                            unbalanced_cids_cnt[cid] = 0
                        unbalanced_cids_cnt[cid] += 1
        
        res_cids = unbalanced_cids - balanced_cids
        res_entries = []
        for cid in res_cids:
            res_entries.append({'cid': cid, 'name': cid_to_name[cid], 'count': unbalanced_cids_cnt[cid]})
        res_entries.sort(key=lambda x: x['count'], reverse=True)

        with open(self.unbalancing_cids_fn, 'w') as f:
            for entry in res_entries:
                f.write(json.dumps(entry) + '\n')
    

    def generate_chems_structures_svg(self):
        for chem in self.chems:
            cid = chem['cid']
            name = chem['cmpdname']
            smiles = chem['smiles']

            svg_fn = os.path.join(self.structures_dir, f"{cid}.svg")
            if os.path.exists(svg_fn):
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)

                drawer = Draw.MolDraw2DSVG(300, 300)  # width, height
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
            except Exception as e:
                self.log(f"Failed to generate structure for '{name}'")
                continue

            with open(os.path.join(self.structures_dir, f"{cid}.svg"), "w") as f:
                f.write(svg)
    

    def __get_mol_organic_mark(self, mol):
        try:
            mol = Chem.AddHs(mol)

            atoms = mol.GetAtoms()
            bonds = mol.GetBonds()
            
            has_carbon = any(atom.GetAtomicNum() == 6 for atom in atoms)
            
            if not has_carbon:
                return False
            
            for bond in bonds:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6):
                    return True
                
                if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 1) or \
                (atom1.GetAtomicNum() == 1 and atom2.GetAtomicNum() == 6):
                    return True
            
            return False
        
        except Exception as e:
            self.log("Error generating organic mark. Assuming True")
            # If complex smiles then likely organic
            return True
    

    def generate_organic_marks_for_chems(self):
        for chem in self.chems:
            smiles = chem['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise Exception(f"Invalid smiles for {chem['cmpdname']}")
            chem['organic'] = self.__get_mol_organic_mark(mol)

        self.__update_chems(self.chems)


    def generate_edges(self):
        reactions = self.__load_jsonl(self.reactions_parsed_balanced_fn)
        
        edge_reaction_id_map = dict()
        for react in reactions:
            react_id = react['rid']
            for r in react['reagents']:
                r_cid = r['cid']
                for p in react['products']:
                    p_cid = p['cid']
                    edge = (r_cid, p_cid)
                    if edge not in edge_reaction_id_map:
                        edge_reaction_id_map[edge] = []
                    edge_reaction_id_map[edge].append(react_id)
        
        with open(self.chems_edges_fn, 'w') as f:
            for edge in edge_reaction_id_map:
                entry = {'first': edge[0], 'second': edge[1], 'reactions': edge_reaction_id_map[edge]}
                f.write(json.dumps(entry) + '\n')
        
        self.log(f"Generated {len(edge_reaction_id_map)} edges")
    


    def __get_mol_fingerprint(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        bitstring = fp.ToBitString()
        popcount = sum([int(x) for x in bitstring])

        chunks = [bitstring[i:i+32] for i in range(0, 1024, 32)]
        ints32 = [int(c, 2) - 2**32 if int(c, 2) >= 2**31 else int(c, 2) for c in chunks]

        return {'bits': ints32, 'popcount': popcount}


    def compute_chems_fingerprints(self):
        for chem in self.chems:
            mol = Chem.MolFromSmiles(chem['smiles'])
            if mol is None:
                raise Exception(f"Invalid smiles for {chem['cmpdname']}")

            chem['ECFP4_fp'] = self.__get_mol_fingerprint(mol)
        
        self.__update_chems(self.chems)
    

    def merge_wiki_chems(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")
        
        cid_wiki_map = dict()
        with open(self.wiki_chems_fn) as f:
            for line in f:
                entry = json.loads(line)
                cid_wiki_map[entry['cid']] = entry['wiki']
        
        with open(self.chems_fn, 'w') as f:
            for chem in self.chems:
                cid = chem['cid']
                if cid in cid_wiki_map:
                    chem['wiki'] = cid_wiki_map[cid]
                else:
                    if 'wiki' in chem:
                        chem.pop('wiki')
                f.write(json.dumps(chem) + '\n')
    

    def __get_mol_bertz_complexity(self, mol):
        return GraphDescriptors.BertzCT(mol)

    

    def compute_chems_bertz_complexity(self):        
        for chem in self.chems:
            mol = Chem.MolFromSmiles(chem['smiles'])
            if mol is None:
                raise Exception(f"Invalid smiles for {chem['cmpdname']}")
            chem['bertz_complexity'] =  self.__get_mol_bertz_complexity(mol)
        
        self.__update_chems(self.chems)
    

    def show_bertz_complexity_diff(self):        
        ct_diff_sum = 0
        ct_sum = 0
        ct_bertz_sum = 0
        ct_cnt = 0
        ct_max_cnt = 500
        for i, chem in enumerate(self.chems):
            ct = chem['complexity']
            bertz_ct = chem['bertz_complexity']
            diff = ct - bertz_ct
            ct_diff_sum += diff
            ct_sum += ct
            ct_bertz_sum += bertz_ct
            ct_cnt += 1
            if ct_cnt == ct_max_cnt:
                av_ct_diff = ct_diff_sum / ct_max_cnt
                av_ct = ct_sum / ct_max_cnt
                av_ct_bertz = ct_bertz_sum / ct_max_cnt
                ct_cnt = ct_diff_sum = ct_sum = ct_bertz_sum = 0
                print(f"{i-ct_max_cnt+1}-{i+1}: av_ct_diff = {av_ct_diff}; av_ct = {av_ct}; av_ct_bertz = {av_ct_bertz}")



    def map_ord_reactions_chems_to_cids(self, ord_reactions_fn):
        unmapped_smiles = dict()
        mapped_count = 0
        overall_count = 0
        processed_rids = set()
        with open(ord_reactions_fn) as f_in, open(self.reactions_parsed_ord_fn, 'w') as f_react, open(self.reactions_details_ord_fn, 'w') as f_det:
            for line in f_in:
                reaction = json.loads(line)
                parse_success = True

                def process_substance(substance, target_list):
                    nonlocal parse_success
                    
                    mol = Chem.MolFromSmiles(substance['smiles'])
                    if not mol:
                        return None
                    inchikey = Chem.MolToInchiKey(mol)
                    chem = self.inchikey_chem_map.get(inchikey)
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    if not chem:
                        if smiles not in unmapped_smiles:
                            unmapped_smiles[smiles] = 0
                        unmapped_smiles[smiles] += 1
                        parse_success = False
                        return None

                    cid = chem['cid']
                    name = chem['cmpdname']
                    norm_name = self.__normalize_chem_name(name)

                    target_list.append({'norm_name': norm_name, 'original_name': name, 'cid': cid})

                parsed_reaction = dict()

                parsed_reaction['reagents'] = []
                for reagent in reaction['reagents']:
                    process_substance(reagent, parsed_reaction['reagents'])
                
                parsed_reaction['products'] = []
                for product in reaction['products']:
                    process_substance(product, parsed_reaction['products'])
                
                parsed_details = dict()
                
                if reaction['solvents']:
                    parsed_details['solvents'] = []
                    for solvent in reaction['solvents']:
                        process_substance(solvent, parsed_details['solvents'])
                else:
                    parsed_details['solvents'] = None
                
                if reaction['catalysts']:
                    parsed_details['catalysts'] = []
                    for solvent in reaction['catalysts']:
                        process_substance(solvent, parsed_details['catalysts'])
                else:
                    parsed_details['catalysts'] = None
                
                parsed_details['provenance'] = reaction['provenance']
                parsed_details["description"] = reaction["description"]

                if parsed_details['provenance'] is None:
                    parsed_details['provenance'] = {'doi': None, 'patent': None}

                def deduplicate_part(entry, part):
                    if entry[part] is not None:
                        unique = set()
                        chems = []
                        for chem in entry[part]:
                            if chem['cid'] not in unique:
                                unique.add(chem['cid'])
                                chems.append(chem)
                        entry[part] = chems
                    else:
                        entry[part] = []

                overall_count += 1
                if parse_success:
                    if parsed_details["description"] is None or len(parsed_details["description"]) < 80:
                        continue

                    deduplicate_part(parsed_details, 'solvents')
                    deduplicate_part(parsed_details, 'catalysts')
                    deduplicate_part(parsed_reaction, 'reagents')
                    deduplicate_part(parsed_reaction, 'products')

                    parsed_reaction = self.__assemble_reaction(parsed_reaction)
                    if parsed_reaction['rid'] in processed_rids:
                        continue

                    processed_rids.add(parsed_reaction['rid'])
                    parsed_reaction["source"] = "ord"
                    parsed_reaction['confidence'] = None

                    parsed_details["rid"] = parsed_reaction['rid']
                    parsed_details["source"] = "ord"
                    f_react.write(json.dumps(parsed_reaction) + '\n')
                    f_det.write(json.dumps(parsed_details) + '\n')
                    
                    mapped_count += 1
                    if mapped_count % 1000 == 0:
                        print(f"Mapped {mapped_count} reactions out of {overall_count}")
        
        unmapped_smiles_list = sorted(list(unmapped_smiles.keys()), key=lambda x: unmapped_smiles[x], reverse=True)
        with open(self.unmapped_smiles_fn, 'w') as f_out:
            text = '\n'.join([f"{x}||{unmapped_smiles[x]}" for x in unmapped_smiles_list])
            f_out.write(text)
        
        print()
        print(f"Unmapped smiles: {len(unmapped_smiles)}")
        print(f"Mapped {mapped_count} reactions out of {overall_count}")
    

    def __get_inchikey_from_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToInchiKey(mol)
        except:
            return None
    

    def fetch_smiles_from_pubchem(self, smiles_fn):
        with open(smiles_fn) as f:
            smiles_list = f.read().strip().split('\n')
        
        blacklist = set()
        if os.path.exists(self.chem_smiles_blacklisted_fn):
            with open(self.chem_smiles_blacklisted_fn) as f:
                blacklist = set(f.read().strip().split('\n'))
        
        with open(self.chems_fn, 'a') as f_out:
            for chem_smiles in smiles_list:
                if chem_smiles in blacklist:
                    continue
    
                inchikey = self.__get_inchikey_from_smiles(chem_smiles)
                if inchikey is None or inchikey in self.inchikey_chem_map:
                    continue

                try:
                    fetched_chems = pcp.get_compounds(chem_smiles, 'smiles')

                    if not fetched_chems:
                        self.log(f"Failed to fetch pubchem data for smiles: '{chem_smiles}'")
                        continue
                    chem = fetched_chems[0]
                    
                    mol_original = Chem.MolFromSmiles(chem_smiles)
                    mol_fetched = Chem.MolFromSmiles(chem.smiles)
                    inchi_original = Chem.MolToInchi(mol_original)
                    inchi_fetched = Chem.MolToInchi(mol_fetched)

                    if inchi_original != inchi_fetched:
                        self.log(f"Fetched substance's inchi doesn't match with original for '{chem_smiles}'. ({inchi_original} != {inchi_fetched})")
                        continue

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

                except Exception as e:
                    self.log(f"Exception during fetching: {e}")
                    continue

                if self.__write_chem_entry(chem_pc_data, f_out):
                    self.log(f"Fetched '{chem_pc_data['cmpdname']}' for unmapped smiles '{chem_smiles}'")
                else:
                    self.log(f"Failed to write '{chem_pc_data['cmpdname']}' for unmapped smiles: '{chem_smiles}'")
    

    def fetch_chems_cids_from_pubchem_f(self, cids_fn):
        with open(cids_fn) as f:
            cids = [int(x) for x in f.read().strip().split('\n')]
        
        self.fetch_chems_cids_from_pubchem(*cids)
    

    def __write_chem_entry(self, entry, f_chem) -> bool:
        if not isinstance(entry['cid'], int):
            return False

        if entry['cid'] in self.cids_blacklist or entry['inchikey'] in self.inchikey_chem_map:
            return False

        f_chem.write(json.dumps(entry) + '\n')
        f_chem.flush()

        return True

    def merge_parsed_files(self, out_fn, *parsed_reactions_files):
        rid_reaction = dict()
        total_reactions = 0
        for fn in parsed_reactions_files:
            reactions = self.__load_jsonl(fn)
            
            total_reactions += len(reactions)
            
            for react in reactions:
                rid = react['rid']
                if rid not in rid_reaction:
                    rid_reaction[rid] = react
                else:
                    old_react = rid_reaction[rid]
                    old_source = old_react.get('source')
                    new_source = react.get('source')
                    new_source_priority = self.sources_priority.get(new_source, -1)
                    old_source_priority = self.sources_priority.get(old_source, -1)
                    if new_source_priority > old_source_priority:
                        rid_reaction[rid] = react

        reactions_res = list(rid_reaction.values())
        self.__write_jsonl(reactions_res, out_fn, backup=False)
        
        print(f"Total reactions number: {total_reactions}; unique reactions written: {len(reactions_res)}")
    

    def merge_reactions(self):
        self.merge_parsed_files(self.reactions_parsed_fn, self.reactions_parsed_llm_fn, self.reactions_parsed_ord_fn)
    
    def merge_details(self):
        self.merge_parsed_files(self.reactions_details_fn, self.reactions_details_llm_fn, self.reactions_details_ord_fn)

    

    def extract_chems_cas_numbers(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        cnt = 0
        for chem in chems:
            cas_numbers = list(map(lambda x: x.strip(), (filter(lambda x: re.fullmatch(r'\d{2,7}-\d\d-\d', x), chem['cmpdsynonym']))))
            cas_numbers = list(set(cas_numbers))
            chem['cas'] = cas_numbers
            if len(cas_numbers) != 0:
                cnt += 1
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')
        
        print(f"Extracted CAS numbers for {cnt} chems")
    

    def get_background_substances(self, k):
        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        cid_chem_map = self.__get_cid_chem_map()
        
        reagents_cids_count = dict()
        for cid in cid_chem_map:
            reagents_cids_count[cid] = 0
        
        for react in reactions:
            for cid in [x['cid'] for x in react['reagents']]:
                reagents_cids_count[cid] += 1
        
        background_cids = sorted(list(reagents_cids_count.keys()), key=lambda x: reagents_cids_count[x], reverse=True)[:k]

        with open(self.background_cids_fn, 'w') as f:
            f.write(json.dumps(background_cids, indent=2))
    

    def get_commonnes_chems_sorting(self):
        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        reagents_cids_count = dict()
        for cid in self.cid_chem_map:
            reagents_cids_count[cid] = 0
        
        for react in reactions:
            for cid in [x['cid'] for x in react['reagents']]:
                reagents_cids_count[cid] += 1
        
        sorted_cids = sorted(list(reagents_cids_count.keys()), key=lambda x: reagents_cids_count[x], reverse=True)

        with open(self.commonness_sorted_cids_fn, 'w') as f:
            f.write(json.dumps(sorted_cids, indent=2))

    
    

    def extract_radicals_list(self, out_fn):
        chems = self.__load_jsonl(self.chems_fn)
        radicals = []
        for chem in chems:
            all_names = chem['cmpdsynonym'] + [chem['cmpdname']]
            for name in all_names:
                if '(.)' in name:
                    radicals.append({'cid': chem['cid'], 'name': chem['cmpdname']})
                    break
        
        self.__write_jsonl(radicals, out_fn, backup=False)
    

    def fetch_elements(self):
        with open(self.elements_fn, 'w') as f:
            base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/TXT"
            for i, el in enumerate(periodictable.elements):
                if i >= 118:
                    break
                name = el.name
                response = requests.get(base_url.format(name))
                if response.status_code != 200:
                    print(f"Failed to obtain cid for '{name}'")
                    continue
                cid = int(response.text.strip())
                if cid not in self.cid_chem_map:
                    print(f"'{name}' not in chems file. Skipping...")
                    continue

                chem = self.cid_chem_map[cid]
                mol = Chem.MolFromInchi(chem['inchi'])
                mol = Chem.AddHs(mol)
                if not mol:
                    print(f"Failed to build mol object for '{name}'")
                    continue
                atom_count = mol.GetNumAtoms()

                entry = {'cid': cid, 'name': name, 'symbol': el.symbol, 'atom_count': atom_count}
                f.write(json.dumps(entry) + '\n')
                f.flush()

                print(f"Fetched '{name}' ({cid}), atom count: {atom_count}")
    


    def compute_reactions_enthalpies(self, out_fn):
        reactions = self.__load_jsonl(self.reactions_parsed_balanced_fn)
        thermo = self.__load_jsonl('data/thermo/chems_thermo_xtb.jsonl')
        cid_to_thermo = {th['cid']: th for th in thermo}

        KCAL_PER_HARTREE = 627.5094740631

        with open(out_fn, 'w') as f:
            for react in reactions:
                if not react['balanced']:
                    continue

                react_str = self.__get_reaction_as_str(react)
                fail = False

                H_r = 0
                for r in react['reagents']:
                    cid = r['cid']
                    if cid not in cid_to_thermo or cid_to_thermo[cid]['Ht_Eh'] is None:
                        fail = True
                        break
                    H_r += r['coeff'] * cid_to_thermo[r['cid']]['Ht_Eh']
                
                H_p = 0
                for p in react['products']:
                    cid = p['cid']
                    if cid not in cid_to_thermo or cid_to_thermo[cid]['Ht_Eh'] is None:
                        fail = True
                        break
                    H_p += p['coeff'] * cid_to_thermo[p['cid']]['Ht_Eh']
                
                if fail:
                    continue
                
                dH = (H_p - H_r) * KCAL_PER_HARTREE

                f.write(json.dumps({'rid': react['rid'], 'dH': dH}) + '\n')
    

    def get_conflicting_synonyms(self, out_fn):
        name_cid_map = dict()
        with open(out_fn, 'w') as f:
            for chem in self.chems:
                cid = chem['cid']
                all_names = [chem['cmpdname']] + chem['cmpdsynonym']
                for i, name in enumerate(all_names):
                    norm_name = self.__normalize_chem_name(name, is_clean=True)
                    if norm_name not in name_cid_map:
                        name_cid_map[norm_name] = (cid, i)
                    else:
                        old_cid, old_cid_i = name_cid_map[norm_name]
                        if old_cid == cid or (old_cid_i > 10 or i > 10):
                            continue
                        old_chem = self.cid_chem_map[old_cid]
                        old_name = old_chem['cmpdname']
                        old_inchi = old_chem['inchi']
                        old_cas = old_chem['cas']
                        old_syn_num = len(old_chem['cmpdsynonym'])
                        curr_name = chem['cmpdname']
                        curr_inchi = chem['inchi']
                        curr_cas = chem['cas']
                        curr_syn_num = len(chem['cmpdsynonym'])
                        inchi_overlap = 'yes' if old_inchi in curr_inchi or curr_inchi in old_inchi else 'no'
                        f.write(f"'{curr_name}' ({cid}) - '{old_name}' ({old_cid}): conflict at '{name}' (normalized: '{norm_name}') (syn. index: {i}/{curr_syn_num} - {old_cid_i}/{old_syn_num})\n")
                        f.write(f"(inchi: '{curr_inchi}' - {old_inchi}(overlap: {inchi_overlap})) (cas: {curr_cas} - {old_cas})\n\n")


    def map_crc_chems_to_cids(self):
        names = [x['name'] for x in self.__load_jsonl('data/crc_handbook/inorganic_constants.jsonl')]
        names += [x['name'] for x in self.__load_jsonl('data/crc_handbook/organic_constants.jsonl')]

        with open('data/crc_unmapped_names.txt', 'w') as f:
            for name in names:
                norm_name = self.__normalize_chem_name(name, is_clean=True)
                if norm_name not in self.name_cid_map:
                    f.write(f"{norm_name}||{name}||0" + '\n')


    

    def __load_jsonl(self, filename):
        with open(filename) as f:
            return [json.loads(x) for x in f.read().strip().split('\n')]
    
    def __write_jsonl(self, entries, filename, backup=True):
        if os.path.exists(filename) and backup:
            shutil.copy(filename, f"{filename}.backup")

        with open(filename, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
    

    def populate_db(self):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        all_cids = set([chem['cid'] for chem in chems])
        
        conn = psycopg2.connect(database=self.db_name)
        cur = conn.cursor()

        sql = \
        "INSERT INTO compounds (cid, name, mf, mw, charge, smiles, inchi, inchikey, complexity, bertz_complexity, organic) " \
        "VALUES %s " \
        "ON CONFLICT (cid) DO NOTHING "

        
        execute_values(cur, sql, [(chem['cid'],
                                    chem['cmpdname'],
                                    chem['mf'],
                                    chem['mw'],
                                    chem['charge'],
                                    chem['smiles'],
                                    chem['inchi'],
                                    chem['inchikey'],
                                    chem['complexity'],
                                    chem['bertz_complexity'],
                                    chem['organic']) for chem in chems])

        sql = \
        "INSERT INTO compound_synonyms (cid, synonym) " \
        "VALUES %s " \
        "ON CONFLICT (cid, synonym) DO NOTHING"
        data = [(chem['cid'], syn) for chem in chems for syn in chem['cmpdsynonym'] if syn]
        execute_values(cur, sql, data)


        sql = \
        "INSERT INTO compound_fingerprints (cid, ECFP4_fp, popcount) " \
        "VALUES %s "
        data = [(chem['cid'], chem['ECFP4_fp']['bits'], chem['ECFP4_fp']['popcount']) for chem in chems]
        execute_values(cur, sql, data)


        sql = \
        "INSERT INTO compound_cas (cid, cas) " \
        "VALUES %s"
        data = [(chem['cid'], cas) for chem in chems for cas in chem['cas'] if 'cas' in chem]
        execute_values(cur, sql, data)


        sql = \
        "INSERT INTO compound_wiki (cid, wiki) " \
        "VALUES %s "
        data = [(x['cid'], x['wiki']) for x in chems if 'wiki' in x]
        execute_values(cur, sql, data)

        with open(self.hazards_chems_fn) as f:
            hazards = [json.loads(x) for x in f.read().strip().split('\n')]
        
        sql = \
        "INSERT INTO compound_nfpa (cid, health, flammability, instability) " \
        "VALUES %s"

        nfpa_data = [(entry['cid'], entry['nfpa'].get('healthHazard'), entry['nfpa'].get('fireHazard'), entry['nfpa'].get('instability')) for entry in hazards if entry['cid'] in all_cids]
        execute_values(cur, sql, nfpa_data)


        sql = \
        "INSERT INTO compound_hazard_statements (cid, statement) " \
        "VALUES %s"
        statements_data = [(entry['cid'], statement) for entry in hazards for statement in entry['statements'] if entry['cid'] in all_cids]
        execute_values(cur, sql, statements_data)

        sql = \
        "INSERT INTO compound_hazard_pictograms (cid, pictogram) " \
        "VALUES %s"
        pictograms_data = [(entry['cid'], pic) for entry in hazards for pic in entry['pictograms'] if entry['cid'] in all_cids]
        execute_values(cur, sql, pictograms_data)

        with open(self.categories_fn) as f:
            categories = [json.loads(x) for x in f.read().strip().split('\n')]
        
        sql = \
        "INSERT INTO categories (code, name, domain) " \
        "VALUES %s"
        categories_data = [(c['code'], c['name'], c['domain']) for c in categories]
        execute_values(cur, sql, categories_data)


        with open(self.chems_categories_fn) as f:
            chems_categories = [json.loads(x) for x in f.read().strip().split('\n')]
        
        sql = \
        "INSERT INTO compound_categories (cid, category_code) " \
        "VALUES %s"
        chems_categories_data = [(entry['cid'], cat) for entry in chems_categories for cat in entry['categories'] if entry['cid'] in all_cids]
        execute_values(cur, sql, chems_categories_data)

        with open(self.chems_descriptions_fn) as f:
            chems_descriptions_data = []
            for x in f.read().strip().split('\n'):
                entry = json.loads(x)
                if entry['cid'] in all_cids:
                    chems_descriptions_data.append((entry['cid'], entry['description']))
        
        sql = \
        "INSERT INTO compound_descriptions (cid, description) " \
        "VALUES %s"
        execute_values(cur, sql, chems_descriptions_data)

        with open(self.background_cids_fn) as f:
            background_cids = [(cid,) for cid in json.loads(f.read())]
        
        sql = \
        "INSERT INTO background_compounds (cid) " \
        "VALUES %s"
        execute_values(cur, sql, background_cids)

        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(self.reactions_details_fn) as f:
            details = [json.loads(x) for x in f.read().strip().split('\n')]
        
        rid_local_id_map = dict()
        for i, react in enumerate(reactions):
            rid_local_id_map[react['rid']] = i+1
        
        rid_details_map = dict()
        for entry in details:
            rid_details_map[entry['rid']] = entry
        
        
        sql = \
        "INSERT INTO reactions (rid, complexity, source, balanced, confidence) " \
        "VALUES %s"
        sql_reactants = \
        "INSERT INTO reaction_reactants (cid, rid) " \
        "VALUES %s"
        sql_products = \
        "INSERT INTO reaction_products (cid, rid) " \
        "VALUES %s"
        sql_solvents = \
        "INSERT INTO reaction_solvents (cid, rid) " \
        "VALUES %s"
        sql_catalysts = \
        "INSERT INTO reaction_catalysts (cid, rid) " \
        "VALUES %s"
        sql_details = \
        "INSERT INTO reaction_details (rid, doi, patent, description, source, confidence) " \
        "VALUES %s"
        execute_values(cur, sql, [(x['rid'], x['complexity'], x['source'], x['balanced'], x['confidence']) for x in reactions])
        execute_values(cur, sql_reactants, [(x['cid'], react['rid']) for react in reactions for x in react['reagents']])
        execute_values(cur, sql_products, [(x['cid'], react['rid']) for react in reactions for x in react['products']])
        execute_values(cur, sql_solvents, [(x['cid'], rid) for rid in rid_details_map for x in rid_details_map[rid]['solvents']  if rid in rid_local_id_map])
        execute_values(cur, sql_catalysts, [(x['cid'], rid) for rid in rid_details_map for x in rid_details_map[rid]['catalysts'] if rid in rid_local_id_map])
        execute_values(cur, sql_details, [(rid, rid_details_map[rid]['provenance']['doi'], rid_details_map[rid]['provenance']['patent'], rid_details_map[rid]['description'], rid_details_map[rid]['source'], rid_details_map[rid]['confidence']) for rid in rid_details_map if rid in rid_local_id_map])

        
        conn.commit()

        cur.close()
        conn.close()



if __name__ == "__main__":
    chemsllm = ChemsLLM("data/", "chemistry")
    #chemsllm.get_raw_reactions(max_workers=20)
    #chemsllm.process_raw_reactions('raw_reactions.jsonl')
    #chemsllm.deduplicate_raw_reactions()
    #chemsllm.validate_raw_reactions(max_workers=20)
    #chemsllm.fix_broken_raw_reactions(max_workers=1)
    #print(chemsllm.find_all_unicode_chars_in_raw_reactions())
    #chemsllm.map_raw_reactions_chems_to_cids()
    #chemsllm.fetch_names_from_pubchem()
    #chemsllm.organize_chems_file()
    #chemsllm.balance_parsed_reactions()
    #chemsllm.find_unbalancing_chems()
    #chemsllm.get_uncommon_raw_reactions_for_wiki_chems(max_workers=20)
    #chemsllm.validate_raw_reactions(raw_reactions_fn="data/top_rare_raw_reactions.jsonl", max_workers=20)
    #chemsllm.generate_chems_structures_svg()
    #chemsllm.get_rare_raw_reactions_for_top_chems(max_workers=20)
    #chemsllm.generate_organic_marks_for_chems()
    #chemsllm.generate_edges()
    #chemsllm.filter_chems_with_invalid_smiles()
    #chemsllm.compute_chems_fingerprints()
    #chemsllm.merge_wiki_chems()
    #chemsllm.make_chems_smiles_canonic()
    #chemsllm.compute_chems_bertz_complexity()
    #chemsllm.show_bertz_complexity_diff()
    #chemsllm.map_ord_reactions_chems_to_cids('cleaned_ord.jsonl')
    #chemsllm.fetch_smiles_from_pubchem()
    #chemsllm.fetch_chems_cids_from_pubchem('cids.txt')
    #chemsllm.merge_parsed_files("data/reactions_parsed/merged_reactions_parsed.jsonl", "data/reactions_parsed/reactions_parsed_ord.jsonl", "data/reactions_parsed/reactions_parsed.jsonl")
    #chemsllm.balance_parsed_reactions()
    #chemsllm.generate_edges()
    #chemsllm.populate_db()
    #chemsllm.deduplicate_chems_rebind_reactions()
    #chemsllm.fix_details()
    #chemsllm.fix_reactions()
    #chemsllm.test()
    #chemsllm.get_uncommon_raw_reactions_for_wiki_chems_products_only(max_workers=20)
    #chemsllm.get_chems_descriptions(max_workers=20)
    #chemsllm.get_reactions_descriptions(max_workers=20)
    #chemsllm.validate_raw_reactions(raw_reactions_fn="data/wiki_products_raw_reactions.jsonl", max_workers=20)
    #chemsllm.clean_data_populate_tables(rehash_required=True)
    #chemsllm.extract_chems_cas_numbers()
    #chemsllm.get_background_substances(20)
    #chemsllm.get_commonnes_chems_sorting()
    #chemsllm.filter_ord_reactions()
    #chemsllm.extract_radicals_list('data/misc/radicals.jsonl')
    #chemsllm.clean_ord_reactions_from_radicals()
    #chemsllm.filter_ord_reactions()
    #chemsllm.fetch_elements()
    #chemsllm.compute_reactions_enthalpies('reaction_enthalpies.jsonl')
    #chemsllm.test()
    #chemsllm.merge_details()
    #chemsllm.merge_reactions()
    #chemsllm.get_conflicting_synonyms('conflict.txt')
    #chemsllm.map_crc_chems_to_cids()
    #chemsllm.fetch_names_from_pubchem('data/crc_unmapped_names.txt')
    chemsllm.get_commonnes_chems_sorting()

    

