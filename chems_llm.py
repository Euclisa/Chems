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
from rdkit.Chem import Draw, AllChem, GraphDescriptors, rdMolDescriptors
import hashlib
import psycopg2
from psycopg2.extras import execute_values
import base64
import random
from time import sleep
import requests
from bs4 import BeautifulSoup

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

        self.raw_reactions_fn = os.path.join(self.data_dir, "raw_reactions.jsonl")
        self.wiki_raw_reactions_fn = os.path.join(self.data_dir, "wiki_raw_reactions.jsonl")
        self.top_rare_raw_reactions_fn = os.path.join(self.data_dir, "top_rare_raw_reactions.jsonl")
        self.raw_reactions_verdict_fn = os.path.join(self.data_dir, "raw_reactions_verdict.jsonl")
        self.raw_reactions_staged_fn = os.path.join(self.data_dir, "raw_reactions_staged.jsonl")
        self.products_wiki_raw_reactions_fn = os.path.join(self.data_dir, 'wiki_products_raw_reactions.jsonl')
        self.reactions_parsed_fn = os.path.join(self.data_dir, "reactions_parsed.jsonl")
        self.reactions_parsed_balanced_fn = os.path.join(self.data_dir, "reactions_parsed_balanced.jsonl")
        self.unmapped_names_fn = os.path.join(self.data_dir, "unmapped_names.txt")
        self.unmapped_names_blacklisted_fn = os.path.join(self.data_dir, "unmapped_names_blacklisted.txt")
        self.unbalancing_cids_fn = os.path.join(self.data_dir, "unbalancing_cids.txt")
        self.chems_fn = os.path.join(self.data_dir, "chems.jsonl")
        self.chems_categories_fn = os.path.join(self.data_dir, "chems_categories.jsonl")
        self.categories_fn = os.path.join(self.data_dir, "categories.jsonl")
        self.wiki_chems_fn = os.path.join(self.data_dir, "wiki_chems.jsonl")
        self.hazards_chems_fn = os.path.join(self.data_dir, "hazards_chems.jsonl")
        self.chems_edges_fn = os.path.join(self.data_dir, 'chems_edges.jsonl')
        self.unmapped_smiles_fn = os.path.join(self.data_dir, 'unmapped_smiles.txt')
        self.unmapped_smiles_blacklisted_fn = os.path.join(self.data_dir, 'unmapped_smiles_blacklisted.txt')
        self.reactions_parsed_ord_fn = os.path.join(self.data_dir, 'reactions_parsed_ord.jsonl')
        self.reactions_parsed_details_ord_fn = os.path.join(self.data_dir, 'reactions_parsed_details_ord.jsonl')
        self.reactions_details_fn = os.path.join(self.data_dir, 'reactions_details.jsonl')
        self.chems_descriptions_fn = os.path.join(self.data_dir, 'chems_descriptions.jsonl')
        self.reactions_descriptions_fn = os.path.join(self.data_dir, 'reactions_descriptions.jsonl')

        self.unmapped_names_delimiter = "||"
        
        self.sources_priority = {
            "ord": 10,
            self.gpt_oss: 5,
            self.qwen: 4,
            self.grok: 3
        }



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
    

    def get_uncommon_raw_reactions_for_wiki_chems(self, max_workers=1):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(self.wiki_chems_fn) as f:
            wiki_chems_cids = set([json.loads(x)['cid'] for x in f.read().strip().split('\n')])
        
        processed = self.__get_processed_entries(self.wiki_raw_reactions_fn, 'cid')

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.wiki_raw_reactions_fn, 'a') as f_out:
            futures = []
            for chem in chems:
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
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
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
            for chem in chems:
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
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed = self.__get_processed_entries(self.products_wiki_raw_reactions_fn, 'cid')

        staged_chems = [chem for chem in chems if 'wiki' in chem and chem['cid'] not in processed]
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

        with open(raw_reactions_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
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
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        for chem in chems:
            chems_power[chem['cid']] = 0

        with open(self.reactions_parsed_balanced_fn) as f:
            for line in f:
                reaction = json.loads(line)
                all_cids = [x['cid'] for x in reaction['reagents']] + [x['cid'] for x in reaction['products']]
                for cid in all_cids:
                    chems_power[cid] += 1
        
        with open(self.hazards_chems_fn) as f:
            hazard_cids = set([json.loads(x)['cid'] for x in f.read().strip().split('\n')])
        
        chems.sort(key=lambda x: chems_power[x['cid']], reverse=True)

        processed = self.__get_processed_entries(self.chems_descriptions_fn, 'cid')

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.chems_descriptions_fn, 'a') as f_out:
            futures = []
            for chem in chems:
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
        reagents_str = ' + '.join([x['original_name'] for x in reaction['reagents']])
        products_str = ' + '.join([x['original_name'] for x in reaction['products']])

        return f"{reagents_str} -> {products_str}"

    def get_reactions_descriptions(self, max_workers=1):
        chems_power = dict()
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        for chem in chems:
            chems_power[chem['cid']] = int('wiki' in chem)

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
    

    def __extract_name_cid_map(self):
        name_cid_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                entry = json.loads(line)
                cid = entry['cid']
                name_cid_map[self.__normalize_chem_name(entry['cmpdname'], is_clean=True)] = cid
                synonyms = list(filter(lambda x: not re.search(r'\d{3,}', x), entry['cmpdsynonym']))
                for syn in synonyms:
                    name_cid_map[self.__normalize_chem_name(syn, is_clean=True)] = cid
        
        return name_cid_map

    
    def __get_reaction_hash(self, reaction):
        reagents_cids = sorted([x['cid'] for x in reaction['reagents']])
        products_cids = sorted([x['cid'] for x in reaction['products']])
        reagents_str = '(' + ','.join([str(x) for x in reagents_cids]) + ')'
        products_str = '(' + ','.join([str(x) for x in products_cids]) + ')'
        reaction_enc = (reagents_str + products_str).encode("utf-8")
        hash_bytes = hashlib.sha256(reaction_enc).digest()
        hash_b64 = base64.b64encode(hash_bytes[:16]).decode("utf-8")

        return hash_b64


    def __get_cid_chem_map(self):
        cid_chem_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                chem = json.loads(line)
                cid_chem_map[chem['cid']] = chem
        
        return cid_chem_map


    def __assemble_reaction_from_participants(self, reagents, products, cid_chem_map: dict):
        max_reagent_complexity = max([cid_chem_map[x['cid']]['complexity'] for x in reagents])
        max_product_complexity = max([cid_chem_map[x['cid']]['complexity'] for x in products])
        for reagent in reagents:
            r_complexity = cid_chem_map[reagent['cid']]['complexity']
            complexity_difference_thr = 20.0
            crucial = r_complexity*2 > max_product_complexity or (max_product_complexity-r_complexity) < complexity_difference_thr or r_complexity*2 > max_reagent_complexity or (max_reagent_complexity-r_complexity) < complexity_difference_thr
            reagent['crucial'] = crucial
        
        av_complexity = 0
        for chem in products:
            av_complexity += cid_chem_map[chem['cid']]['complexity']
        
        for chem in reagents:
            av_complexity += cid_chem_map[chem['cid']]['complexity']
        av_complexity /= len(products) + len(reagents)

        reaction = {'reagents': reagents, 'products': products}
        reaction['complexity'] = av_complexity
        
        reaction_hash = self.__get_reaction_hash(reaction)
        reaction['rid'] = reaction_hash

        return reaction

    
    def __parse_reaction_str(self, reaction_str: str, name_cid_map: dict, cid_chem_map: dict):
        reactions_str_split = reaction_str.split('->')
        if len(reactions_str_split) != 2:
            return None
        
        reagents, products = reactions_str_split

        parse_success = True

        unmapped_names = set()

        reagents_clean = []
        reagents_cids = set()
        for chem_name in reagents.split('+'):
            norm_name = self.__normalize_chem_name(chem_name)
            if norm_name in {"light", "heat", "catalyst"}:
                continue
            clean_name = self.__clean_chem_name(chem_name)
            cid = name_cid_map.get(norm_name)
            if cid is None:
                parse_success = False
                unmapped_names.add(clean_name)

            if cid not in reagents_cids or cid is None:
                reagents_clean.append({'norm_name': norm_name, 'original_name': clean_name, 'cid': cid})
                reagents_cids.add(cid)
        
        reagents_clean = list({d["cid"]: d for d in reagents_clean}.values())
        
        products_clean = []
        products_cids = set()
        for chem_name in products.split('+'):
            norm_name = self.__normalize_chem_name(chem_name)
            if norm_name in {"otherproducts"}:
                continue
            clean_name = self.__clean_chem_name(chem_name)
            cid = name_cid_map.get(norm_name)
            if cid is None:
                parse_success = False
                unmapped_names.add(clean_name)
    
            if (cid not in products_cids or cid is None) and (cid not in reagents_cids or cid is None):
                products_clean.append({'norm_name': norm_name, 'original_name': clean_name, 'cid': cid})
                products_cids.add(cid)
        
        products_clean = list({d["cid"]: d for d in products_clean}.values())

        if products_cids & reagents_cids or not products_clean:
            parse_success = False
        
        if not parse_success:
            return parse_success, None, unmapped_names
        
        reaction = self.__assemble_reaction_from_participants(reagents_clean, products_clean, cid_chem_map)

        return parse_success, reaction, unmapped_names


    def map_raw_reactions_chems_to_cids(self):
        cid_chem_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                chem = json.loads(line)
                cid = chem['cid']
                cid_chem_map[cid] = chem

        name_cid_map = self.__extract_name_cid_map()
        self.log(f"Name-CID map size: {len(name_cid_map)}")

        unmapped_names = dict()

        parsed = []
        processed_reactions_ids = set()
        with open(self.raw_reactions_verdict_fn) as f:
            for line in f:
                entry = json.loads(line)
                valid = entry['valid']
                reaction = entry['reaction']
                if 'confidence' not in entry:
                    continue
                confidence = entry['confidence']
                if confidence < 0.4:
                    continue
                parse_res = self.__parse_reaction_str(reaction, name_cid_map, cid_chem_map)
                if parse_res is None:
                    continue
                parse_success, parsed_reaction, unmapped_names_curr = parse_res
                for name in unmapped_names_curr:
                    norm_name = self.__normalize_chem_name(name)
                    if norm_name not in unmapped_names:
                        unmapped_names[norm_name] = [0, name]
                    unmapped_names[norm_name][0] += 1
                if not parse_success:
                    continue
                
                reagents_cids = tuple(sorted([x['cid'] for x in parsed_reaction['reagents']]))
                products_cids = tuple(sorted([x['cid'] for x in parsed_reaction['products']]))
                reaction_id = (reagents_cids, products_cids)
                if reaction_id in processed_reactions_ids:
                    continue
                processed_reactions_ids.add(reaction_id)
                
                parsed_reaction['confidence'] = confidence
                parsed_reaction['source'] = entry['source']
                parsed.append(parsed_reaction)
        
        with open(self.reactions_parsed_fn, 'w') as f:
            for react in parsed:
                f.write(json.dumps(react) + '\n')
        
        unmapped_names_list = list(unmapped_names.keys())
        unmapped_names_list.sort(key=lambda x: unmapped_names[x][0], reverse=True)
        with open(self.unmapped_names_fn, 'w') as f:
            for chem_name in unmapped_names_list:
                f.write(f"{chem_name}{self.unmapped_names_delimiter}{unmapped_names[chem_name][1]}{self.unmapped_names_delimiter}{unmapped_names[chem_name][0]}\n")
        
        self.log(f"Successfully parsed {len(parsed)} reactions; unmapped names size: {len(unmapped_names)}")

            
    def fetch_unmapped_names_from_pubchem(self):
        with open(self.unmapped_names_fn) as f:
            unmapped_names = f.read().strip().split('\n')
        
        blacklist = set()
        if os.path.exists(self.unmapped_names_blacklisted_fn):
            with open(self.unmapped_names_blacklisted_fn) as f:
                blacklist = set(map(lambda x: self.__normalize_chem_name(x), f.read().strip().split('\n')))
        
        with open(self.chems_fn, 'a') as f_out, open(self.unmapped_names_blacklisted_fn, 'a') as f_out_black:
            for entry in unmapped_names:
                chem_name_norm, chem_name, cnt = entry.split(self.unmapped_names_delimiter)
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
                if chem:
                    try:
                        # Workaround to prevent fetching gibberish
                        all_names = chem.synonyms + [chem.iupac_name]
                        all_names = [self.__normalize_chem_name(x,is_clean=True) for x in all_names]
                        if any([x in blacklist for x in all_names]):
                            continue
                        if not any([x == chem_name_norm for x in all_names]):
                            f_out_black.write(chem_name + '\n')
                            f_out_black.flush()
                            continue
                    except Exception as e:
                        f_out_black.write(chem_name + '\n')
                        f_out_black.flush()
                        continue

                    f_out.write(json.dumps(chem_pc_data) + '\n')
                    f_out.flush()
                    self.log(f"Fetched '{chem_pc_data['cmpdname']}' for unmapped name '{chem_name}'")
        
    
    def fetch_chems_cids_from_pubchem(self, *args):
        with open(self.chems_fn, 'a') as f_out:
            for cid in args:
                fetched_chems = pcp.get_compounds(cid, 'cid')
                if not fetched_chems:
                    self.log(f"Failed to fetch pubchem data for cid '{cid}'")
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

                f_out.write(json.dumps(chem_pc_data) + '\n')
                f_out.flush()
                self.log(f"Fetched '{chem_pc_data['cmpdname']}' for cid '{cid}'")



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
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

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

        cids_to_discard = self.__make_chems_smiles_canonic(unique_chems)
        unique_chems = list(filter(lambda x: x['cid'] not in cids_to_discard, unique_chems))

        with open(self.chems_fn, 'w') as f:
            for chem in unique_chems:
                f.write(json.dumps(chem) + '\n')
    

    def balance_parsed_reactions(self, reactions_parsed_fn=None):
        if reactions_parsed_fn is None:
            reactions_parsed_fn = self.reactions_parsed_fn
        cid_to_mf = dict()
        with open(self.chems_fn) as f:
            for line in f:
                chem = json.loads(line)
                cid_to_mf[chem['cid']] = chem['mf']
            
        with open(reactions_parsed_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed_reactions = []
        balanced_cnt = 0
        for react in reactions:
            reagents = [cid_to_mf[x['cid']] for x in react['reagents']]
            products = [cid_to_mf[x['cid']] for x in react['products']]
    
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
                mf = cid_to_mf[chem['cid']]
                chem['coeff'] = int(reagents_coeffs[mf])
            
            for chem in react['products']:
                mf = cid_to_mf[chem['cid']]
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
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        for chem in chems:
            cid = chem['cid']
            name = chem['cmpdname']
            smiles = chem['smiles']

            try:
                mol = Chem.MolFromSmiles(smiles)

                drawer = Draw.MolDraw2DSVG(300, 300)  # width, height
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
            except Exception as e:
                print(f"Failed to generate structure for '{name}'")
                continue

            with open(os.path.join(self.structures_dir, f"{cid}.svg"), "w") as f:
                f.write(svg)
    

    def generate_organic_marks_for_chems(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        def is_organic(chem):
            try:
                smiles = chem['smiles']
                mol = Chem.MolFromSmiles(smiles)
                
                if mol is None:
                    raise ValueError("Неверный формат SMILES")

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
                print(f"Error processing smiles SMILES: {e}")
                # If complex smiles then likely organic
                return True
        
        for chem in chems:
            chem['organic'] = is_organic(chem)
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')


    def generate_edges(self):
        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        edge_reaction_id_map = dict()
        for react in reactions:
            react_id = react['rid']
            for r in react['reagents']:
                r_cid = r['cid']
                if not r['crucial']:
                    continue
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
        
        print(f"Generated {len(edge_reaction_id_map)} edges")
    

    def filter_chems_with_invalid_smiles(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(self.unmapped_names_blacklisted_fn, 'a') as f_black:
            filtred_chems = []
            for chem in chems:
                mol = Chem.MolFromSmiles(chem['smiles'])
                if mol is None:
                    print(f"Discarding {chem['cmpdname']}")
                    f_black.write(chem['cmpdname'] + '\n')
                    continue
                filtred_chems.append(chem)

        with open(self.chems_fn, 'w') as f:
            for chem in filtred_chems:
                f.write(json.dumps(chem) + '\n')
        
        print(f"Discarded {len(chems)-len(filtred_chems)} substances")
    

    def compute_chems_fingerprints(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        for chem in chems:
            mol = Chem.MolFromSmiles(chem['smiles'])
            if mol is None:
                raise Exception(f"Invalid smiles for {chem['cmpdname']}")

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            bitstring = fp.ToBitString()
            popcount = sum([int(x) for x in bitstring])

            chunks = [bitstring[i:i+32] for i in range(0, 1024, 32)]
            ints32 = [int(c, 2) - 2**32 if int(c, 2) >= 2**31 else int(c, 2) for c in chunks]

            chem['ECFP4_fp'] = {'bits': ints32, 'popcount': popcount}
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')
    

    def merge_wiki_chems(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        cid_wiki_map = dict()
        with open(self.wiki_chems_fn) as f:
            for line in f:
                entry = json.loads(line)
                cid_wiki_map[entry['cid']] = entry['wiki']
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                cid = chem['cid']
                if cid in cid_wiki_map:
                    chem['wiki'] = cid_wiki_map[cid]
                else:
                    if 'wiki' in chem:
                        chem.pop('wiki')
                f.write(json.dumps(chem) + '\n')
    

    def __make_chems_smiles_canonic(self, chems):
        cids_to_discard = set()
        for chem in chems:
            chem['inchi'] = chem['inchi'].strip()
            try:
                mol = Chem.MolFromInchi(chem['inchi'])
                chem['smiles'] = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"Discarding '{chem['cmpdname']}'")
                cids_to_discard.add(chem['cid'])
        
        return cids_to_discard

    

    def make_chems_smiles_canonic(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        cids_to_discard = self.__make_chems_smiles_canonic(chems)
        chems = list(filter(lambda x: x['cid'] not in cids_to_discard, chems))
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')
        

    

    def compute_chems_bertz_complexity(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        for chem in chems:
            mol = Chem.MolFromSmiles(chem['smiles'])
            chem['bertz_complexity'] =  GraphDescriptors.BertzCT(mol)
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')
    

    def show_bertz_complexity_diff(self):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        ct_diff_sum = 0
        ct_sum = 0
        ct_bertz_sum = 0
        ct_cnt = 0
        ct_max_cnt = 500
        for i, chem in enumerate(chems):
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
    

    def __create_smiles_chem_map(self):
        smiles_chem_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                chem = json.loads(line)
                if 'smiles' not in chem:
                    continue
                smiles_chem_map[chem['smiles']] = chem
        
        return smiles_chem_map

    def __create_inchi_chem_map(self):
        inchi_chem_map = dict()
        with open(self.chems_fn) as f:
            for line in f:
                chem = json.loads(line)
                if 'inchi' not in chem:
                    continue
                inchi_chem_map[chem['inchi']] = chem
        
        return inchi_chem_map


    def map_ord_reactions_chems_to_cids(self, ord_reactions_fn):
        with open(self.reactions_parsed_ord_fn, 'w'):
            pass
        with open(self.reactions_parsed_details_ord_fn, 'w'):
            pass

        smiles_chem_map = self.__create_smiles_chem_map()
        inchi_chem_map = self.__create_inchi_chem_map()
        cid_chem_map = self.__get_cid_chem_map()
        unmapped_smiles = dict()
        mapped_count = 0
        overall_count = 0
        processed_rids = set()
        with open(ord_reactions_fn) as f:
            for line in f:
                reaction = json.loads(line)
                parse_success = True
                reagents = []
                products = []

                def process_substance(substance, target_list):
                    nonlocal parse_success
                    
                    mol = Chem.MolFromSmiles(substance['smiles'])
                    smiles = Chem.MolToSmiles(mol)
                    chem = smiles_chem_map.get(smiles)
                    if not chem:
                        inchi = Chem.MolToInchi(mol)
                        chem = inchi_chem_map.get(inchi)
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

                for reagent in reaction['reagents']:
                    process_substance(reagent, reagents)
                
                for product in reaction['products']:
                    process_substance(product, products)
                
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

                overall_count += 1
                if parse_success:
                    parsed_reaction = self.__assemble_reaction_from_participants(reagents, products, cid_chem_map)
                    if parsed_reaction['rid'] in processed_rids:
                        continue
                    processed_rids.add(parsed_reaction['rid'])
                    parsed_reaction["source"] = "ord"
                    parsed_reaction['confidence'] = None

                    parsed_details["rid"] = parsed_reaction['rid']
                    with open(self.reactions_parsed_ord_fn, 'a') as f_out:
                        f_out.write(json.dumps(parsed_reaction) + '\n')
                    with open(self.reactions_parsed_details_ord_fn, 'a') as f_out:
                        f_out.write(json.dumps(parsed_details) + '\n')
                    
                    mapped_count += 1
                    if mapped_count % 1000 == 0:
                        print(f"Mapped {mapped_count} reactions out of {overall_count}")
        
        unmapped_smiles_list = sorted(list(unmapped_smiles.keys()), key=lambda x: unmapped_smiles[x], reverse=True)
        with open(self.unmapped_smiles_fn, 'w') as f_out:
            text = '\n'.join([f"{x} -> {unmapped_smiles[x]}" for x in unmapped_smiles_list])
            f_out.write(text)
        
        print()
        print(f"Unmapped smiles: {len(unmapped_smiles)}")
        print(f"Mapped {mapped_count} reactions out of {overall_count}")
    

    def fetch_unmapped_smiles_from_pubchem(self):
        with open(self.unmapped_smiles_fn) as f:
            unmapped_smiles = f.read().strip().split('\n')
        
        blacklist = set()
        if os.path.exists(self.unmapped_smiles_blacklisted_fn):
            with open(self.unmapped_smiles_blacklisted_fn) as f:
                blacklist = set(f.read().strip().split('\n'))
        
        with open(self.chems_fn, 'a') as f_out, open(self.unmapped_smiles_blacklisted_fn, 'a') as f_out_black:
            for chem_smiles in unmapped_smiles:
                if chem_smiles in blacklist:
                    continue
                try:
                    fetched_chems = pcp.get_compounds(chem_smiles, 'smiles')

                    if not fetched_chems:
                        f_out_black.write(chem_smiles + '\n')
                        f_out_black.flush()
                        self.log(f"Failed to fetch pubchem data for unmapped smiles '{chem_smiles}'")
                        continue
                    chem = fetched_chems[0]
                    
                    mol_original = Chem.MolFromSmiles(chem_smiles)
                    mol_fetched = Chem.MolFromSmiles(chem.smiles)
                    formula_original = rdMolDescriptors.CalcMolFormula(mol_original)
                    formula_fetched = rdMolDescriptors.CalcMolFormula(mol_fetched)

                    if formula_original != formula_fetched:
                        f_out_black.write(chem_smiles + '\n')
                        f_out_black.flush()
                        self.log(f"Fetched substance's formula doesn't match with original for '{chem_smiles}'. ({formula_original} != {formula_fetched})")
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

                f_out.write(json.dumps(chem_pc_data) + '\n')
                f_out.flush()
                self.log(f"Fetched '{chem_pc_data['cmpdname']}' for unmapped smiles '{chem_smiles}'")
    

    def fetch_chems_cids_from_pubchem_f(self, cids_fn):
        with open(cids_fn) as f:
            cids = [int(x) for x in f.read().strip().split('\n')]
        
        with open(self.chems_fn, 'a') as f_out:
            for cid in cids:
                try:
                    fetched_chems = pcp.get_compounds(cid, 'cid')
                except Exception as e:
                    self.log(f"Exception during fetching {cid}: {e}")
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

                f_out.write(json.dumps(chem_pc_data) + '\n')
                f_out.flush()
                self.log(f"Fetched '{chem_pc_data['cmpdname']}' for cid '{cid}'")


    def merge_parsed_reactions_files(self, out_fn, *parsed_reactions_files):
        processed_rids = set()
        rid_to_index_map = dict()
        reactions_res = []
        total_reactions = 0
        for fn in parsed_reactions_files:
            with open(fn) as f:
                reactions = [json.loads(x) for x in f.read().strip().split('\n')]
            
            total_reactions += len(reactions)
            
            for react in reactions:
                rid = react['rid']
                if rid not in processed_rids:
                    rid_to_index_map[rid] = len(reactions_res)
                    reactions_res.append(react)
                    processed_rids.add(rid)
                else:
                    index = rid_to_index_map[rid]
                    old_react = reactions_res[index]
                    old_source = old_react.get('source')
                    new_source = react.get('source')
                    new_source_priority = self.sources_priority.get(new_source, -1)
                    old_source_priority = self.sources_priority.get(old_source, -1)
                    if new_source_priority > old_source_priority:
                        reactions_res[index] = react
        
        with open(out_fn, 'w') as f:
            for react in reactions_res:
                f.write(json.dumps(react) + '\n')
        
        print(f"Total reactions number: {total_reactions}; unique reactions written: {len(reactions_res)}")
    

    def fix_details(self):
        shutil.copy(self.reactions_details_fn, f"{self.reactions_details_fn}.backup")

        with open(self.reactions_details_fn) as f:
            details = [json.loads(x) for x in f.read().strip().split('\n')]
        
        def fix_part(entry, part):
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
        
        for entry in details:
            fix_part(entry, 'solvents')
            fix_part(entry, 'catalysts')
            
            if entry['provenance'] is None:
                entry['provenance'] = {'doi': None, 'patent': None}
        
        with open(self.reactions_details_fn, 'w') as f:
            for entry in details:
                f.write(json.dumps(entry) + '\n')


    def fix_reactions(self):
        shutil.copy(self.reactions_parsed_balanced_fn, f"{self.reactions_parsed_balanced_fn}.backup")

        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        def fix_part(entry, part):
            unique = set()
            chems = []
            for chem in entry[part]:
                if chem['cid'] not in unique:
                    unique.add(chem['cid'])
                    chems.append(chem)
            entry[part] = chems

        for entry in reactions:
            fix_part(entry, 'reagents')
            fix_part(entry, 'products')
        
        with open(self.reactions_parsed_balanced_fn, 'w') as f:
            for entry in reactions:
                f.write(json.dumps(entry) + '\n')
        
        # Remove duplicates taking into account source priorities
        self.merge_parsed_reactions_files(self.reactions_parsed_balanced_fn, self.reactions_parsed_balanced_fn)
    

    def fix_hazards(self):
        shutil.copy(self.hazards_chems_fn, f"{self.hazards_chems_fn}.backup")

        with open(self.hazards_chems_fn) as f:
            hazards = [json.loads(x) for x in f.read().strip().split('\n')]

        unique = set()
        new_hazards = []
        for entry in hazards:
            if entry['cid'] not in unique:
                unique.add(entry['cid'])
                new_hazards.append(entry)
            entry['statements'] = list(set(entry['statements']))
            entry['pictograms'] = list(set(entry['pictograms'])) 
        
        with open(self.hazards_chems_fn, 'w') as f:
            for entry in new_hazards:
                f.write(json.dumps(entry) + '\n')
    

    def rehash_reactions(self, out_rid_map_fn):
        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        rid_map = []
        for react in reactions:
            old_rid = react['rid']
            react['rid'] = self.__get_reaction_hash(react)
            rid_map.append((old_rid, react['rid']))
        
        with open(out_rid_map_fn, 'w') as f:
            for entry in rid_map:
                f.write(f"{entry[0]} {entry[1]}\n")
    

    def replace_old_rids(self, in_file, rid_map_fn):
        with open(in_file) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(rid_map_fn) as f:
            rid_map = [x.split() for x in f.read().strip().split('\n')]
            rid_map = {old: new for old, new in rid_map}
        
        orphaned_entries_i = set()
        for i, entry in enumerate(entries):
            new_rid = rid_map.get(entry['rid'])
            if new_rid:
                entry['rid'] = new_rid
            else:
                orphaned_entries_i.add(i)
                self.log(f"Reaction with '{entry['rid']}' from file '{in_file}' does not exist")
        

        with open(in_file, 'w') as f:
            for i, entry in enumerate(entries):
                if i not in orphaned_entries_i:
                    f.write(json.dumps(entry) + '\n')
        
        print(f"Successfully rehashed '{in_file}'")

    

    def deduplicate_chems_rebind_reactions(self, out_cid_map_fn):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")
        shutil.copy(self.reactions_parsed_balanced_fn, f"{self.reactions_parsed_balanced_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(self.reactions_parsed_balanced_fn) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        def get_cid_index_map(entries, parts):
            cid_i_map = dict()
            for i, entry in enumerate(entries):
                cids = set([x['cid'] for part in parts for x in entry[part]])
                for cid in cids:
                    if cid not in cid_i_map:
                        cid_i_map[cid] = set()
                    cid_i_map[cid].add(i)
            
            return cid_i_map
        
        cid_reaction_i_map = get_cid_index_map(reactions, ['reagents', 'products'])

        new_chems = dict()
        unique_inchikeys = dict()
        cid_merge_map = []
        for i, chem in enumerate(chems):
            inchikey = chem['inchikey']
            curr_cid = chem['cid']
            if inchikey not in unique_inchikeys:
                unique_inchikeys[inchikey] = i
                new_chems[curr_cid] = chem
            else:
                old_i = unique_inchikeys[inchikey]
                old_cid = chems[old_i]['cid']
                
                old_cid_reactions = cid_reaction_i_map.get(old_cid, set())
                curr_cid_reactions = cid_reaction_i_map.get(curr_cid, set())
                
                def update_cid(entries, reaction_indices, old_value, new_value, parts):
                    for i in reaction_indices:
                        for part in parts:
                            for chem in entries[i][part]:
                                if chem['cid'] == old_value:
                                    chem['cid'] = new_value   

                if len(old_cid_reactions) > len(curr_cid_reactions):
                    update_cid(reactions, curr_cid_reactions, curr_cid, old_cid, ['reagents', 'products'])
                    cid_merge_map.append((curr_cid, old_cid))
                    print(f"Merged {chem['cmpdname']}({curr_cid}) -> {chems[old_i]['cmpdname']}({old_cid}); {len(curr_cid_reactions)} reactions fixed")
                else:
                    update_cid(reactions, old_cid_reactions, old_cid, curr_cid, ['reagents', 'products'])
                    new_chems.pop(old_cid)
                    new_chems[curr_cid] = chem
                    cid_merge_map.append((old_cid, curr_cid))
                    print(f"Merged {chem['cmpdname']}({curr_cid}) <- {chems[old_i]['cmpdname']}({old_cid}); {len(old_cid_reactions)} reactions fixed")


        with open(self.reactions_parsed_balanced_fn, 'w') as f:
            for react in reactions:
                f.write(json.dumps(react) + '\n')
        
        with open(self.chems_fn, 'w') as f:
            for chem in new_chems.values():
                f.write(json.dumps(chem) + '\n')
        
        with open(out_cid_map_fn, 'w') as f:
            for old_cid, new_cid in cid_merge_map:
                f.write(f"{old_cid} {new_cid}\n")
    

    def replace_old_cids(self, in_file, parts, cid_map_fn):
        with open(in_file) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open(cid_map_fn) as f:
            cid_map = [list(map(int, x.split())) for x in f.read().strip().split('\n')]
            cid_map = {old: new for old, new in cid_map}
        
        cnt = 0
        for entry in entries:
            if parts:
                for part in parts:
                    for part_entry in entry[part]:
                        curr_cid = part_entry['cid']
                        if curr_cid in cid_map:
                            part_entry['cid'] = cid_map[curr_cid]
                            cnt += 1
            else:
                curr_cid = entry['cid']
                if curr_cid in cid_map:
                    entry['cid'] = cid_map[curr_cid]
                    cnt += 1
        
        with open(in_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Fixed {cnt} cids in '{in_file}'")
    

    def extract_chems_cas_numbers(self):
        shutil.copy(self.chems_fn, f"{self.chems_fn}.backup")

        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        cnt = 0
        for chem in chems:
            cas_numbers = list(map(lambda x: x.strip(), (filter(lambda x: re.fullmatch(r'\d{2,7}-\d\d-\d', x), chem['cmpdsynonym']))))
            if len(cas_numbers) == 1:
                chem['cas'] = cas_numbers[0]
                cnt += 1
        
        with open(self.chems_fn, 'w') as f:
            for chem in chems:
                f.write(json.dumps(chem) + '\n')
        
        print(f"Extracted CAS numbers for {cnt} chems")
            
    

    def test(self):
        with open(self.chems_fn) as f:
            chems = [json.loads(x) for x in f.read().strip().split('\n')]
        
        with open('cids.txt', 'w') as f:
            for chem in chems:
                f.write(f"{chem['cid']}\n")

    

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
        data = [(chem['cid'], chem['cas']) for chem in chems if 'cas' in chem]
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
        chems_categories_data = [(entry['cid'], cat) for entry in chems_categories for cat in entry['categories']]
        execute_values(cur, sql, chems_categories_data)

        with open(self.chems_descriptions_fn) as f:
            chems_descriptions_data = []
            for x in f.read().strip().split('\n'):
                entry = json.loads(x)
                chems_descriptions_data.append((entry['cid'], entry['description']))
        
        sql = \
        "INSERT INTO compound_descriptions (cid, description) " \
        "VALUES %s"
        execute_values(cur, sql, chems_descriptions_data)

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
        execute_values(cur, sql_solvents, [(x['cid'], rid) for rid in rid_details_map for x in rid_details_map[rid]['solvents']])
        execute_values(cur, sql_catalysts, [(x['cid'], rid) for rid in rid_details_map for x in rid_details_map[rid]['catalysts']])
        execute_values(cur, sql_details, [(rid, rid_details_map[rid]['provenance']['doi'], rid_details_map[rid]['provenance']['patent'], rid_details_map[rid]['description'], rid_details_map[rid]['source'], rid_details_map[rid]['confidence']) for rid in rid_details_map])

        
        conn.commit()

        cur.close()
        conn.close()


    def clean_data_populate_tables(self, rehash_required=False):
        chemsllm.compute_chems_bertz_complexity()
        chemsllm.compute_chems_fingerprints()
        chemsllm.generate_organic_marks_for_chems()
        chemsllm.merge_wiki_chems()
        chemsllm.extract_chems_cas_numbers()
        chemsllm.organize_chems_file()
    
        chemsllm.fix_reactions()
        chemsllm.fix_details()
        chemsllm.fix_hazards()

        cid_map_fn = 'cid_map.txt'
        if not os.path.exists(cid_map_fn):
            print(f"'{cid_map_fn}' wasn't found. Building new...")
            chemsllm.deduplicate_chems_rebind_reactions(cid_map_fn)

        chemsllm.replace_old_cids(self.hazards_chems_fn, [], cid_map_fn)
        chemsllm.replace_old_cids('data/merged_reactions_parsed.jsonl', ['reagents', 'products'], cid_map_fn)
        chemsllm.replace_old_cids(self.raw_reactions_verdict_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.raw_reactions_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.top_rare_raw_reactions_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.wiki_raw_reactions_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.reactions_parsed_details_ord_fn, ['solvents', 'catalysts'], cid_map_fn)
        chemsllm.replace_old_cids(self.reactions_parsed_ord_fn, ['reagents', 'products'], cid_map_fn)
        chemsllm.replace_old_cids(self.reactions_parsed_fn, ['reagents', 'products'], cid_map_fn)
        chemsllm.replace_old_cids(self.wiki_chems_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.reactions_details_fn, ['solvents', 'catalysts'], cid_map_fn)
        chemsllm.replace_old_cids(self.products_wiki_raw_reactions_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.chems_categories_fn, [], cid_map_fn)
        chemsllm.replace_old_cids(self.chems_descriptions_fn, [], cid_map_fn)
    
        if rehash_required:
            rid_map_fn = 'rid_map.txt'
            if not os.path.exists(rid_map_fn):
                print(f"'{rid_map_fn}' wasn't found. Building new...")
                chemsllm.rehash_reactions(rid_map_fn)

            chemsllm.replace_old_rids(self.reactions_parsed_details_ord_fn, rid_map_fn)
            chemsllm.replace_old_rids(self.reactions_details_fn, rid_map_fn)
            chemsllm.replace_old_rids(self.reactions_descriptions_fn, rid_map_fn)
            chemsllm.replace_old_rids(self.reactions_parsed_ord_fn, rid_map_fn)
            chemsllm.replace_old_rids(self.reactions_parsed_fn, rid_map_fn)
            chemsllm.replace_old_rids('data/merged_reactions_parsed.jsonl', rid_map_fn)
            chemsllm.replace_old_rids(self.reactions_parsed_balanced_fn, rid_map_fn)

        chemsllm.fix_reactions()

        chemsllm.populate_db()



                    







if __name__ == "__main__":
    chemsllm = ChemsLLM("data/", "chemistry")
    #chemsllm.get_raw_reactions(max_workers=20)
    #chemsllm.process_raw_reactions('raw_reactions.jsonl')
    #chemsllm.deduplicate_raw_reactions()
    #chemsllm.validate_raw_reactions(max_workers=20)
    #chemsllm.fix_broken_raw_reactions(max_workers=1)
    #print(chemsllm.find_all_unicode_chars_in_raw_reactions())
    #chemsllm.map_raw_reactions_chems_to_cids()
    #chemsllm.fetch_unmapped_names_from_pubchem()
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
    #chemsllm.fetch_unmapped_smiles_from_pubchem()
    #chemsllm.fetch_chems_cids_from_pubchem('cids.txt')
    #chemsllm.merge_parsed_reactions_files("data/merged_reactions_parsed.jsonl", "data/reactions_parsed_ord.jsonl", "data/reactions_parsed.jsonl")
    #chemsllm.balance_parsed_reactions("data/merged_reactions_parsed.jsonl")
    chemsllm.populate_db()
    #chemsllm.deduplicate_chems_rebind_reactions()
    #chemsllm.fix_details()
    #chemsllm.fix_reactions()
    #chemsllm.test()
    #chemsllm.get_uncommon_raw_reactions_for_wiki_chems_products_only(max_workers=20)
    #chemsllm.get_chems_descriptions(max_workers=20)
    #chemsllm.get_reactions_descriptions(max_workers=20)
    #chemsllm.validate_raw_reactions(raw_reactions_fn="data/wiki_products_raw_reactions.jsonl", max_workers=20)
    #chemsllm.merge_parsed_reactions_files("reactions_descriptions.jsonl", "data/reactions_parsed_details_ord.jsonl", "reactions_descriptions.jsonl")
    #chemsllm.clean_data_populate_tables(rehash_required=True)
    #chemsllm.extract_chems_cas_numbers()

    

