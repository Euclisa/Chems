from openai import OpenAI
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class ChemsLLM:
    def __init__(self, chems_fn, data_dir, api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        self.completion_tokens_total = 0
        self.input_tokens_total = 0
        self.tokens_total_lock = threading.Lock()
        
        self.print_lock = threading.Lock()

        self.chems_fn = chems_fn

        self.gpt_oss = "openai/gpt-oss-120b"

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.raw_reactions_fn = os.path.join(self.data_dir, "raw_reactions.jsonl")
        self.raw_reactions_verdict_fn = os.path.join(self.data_dir, "raw_reactions_verdict.jsonl")
        self.raw_reactions_staged_fn = os.path.join(self.data_dir, "raw_reactions_staged.jsonl")



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
            "Please, review the provided reactions. Identify any erroneous reactions and correct them where possible. Return revised initial reactions list that comply with the initial requirements."

            instruct_validate = \
            "You will be given a list of unbalanced chemical reaction schemes. " \
            "For each scheme, determine if the reaction is chemically possible and whether the listed reactants and products are correct. " \
            "If the reaction is valid, output only 'Valid'. If it is not valid, output only 'Invalid'. " \
            "Print one result per line and do not include any additional text."


            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instruct}
            ]

            response = self.__fetch_llm_answer(messages, self.gpt_oss)
            reactions = self.__get_reactions_from_response(response)
            if not reactions:
                self.log(f"Rejected '{chem_name}'")
                return None
            
            #self.log(f"Got {len(reactions)} initial reactions for {chem_name}")
            
            response = '\n'.join(reactions)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": instruct_revalidate})

            response = self.__fetch_llm_answer(messages, self.gpt_oss)
            reactions_revalid = self.__get_reactions_from_response(response)
            if not reactions_revalid:
                self.log(f"Failed to fetch revalidated reactions for '{chem_name}'. Assuming all are valid...")
                reactions_revalid = reactions
            
            self.log(f"Got {len(reactions)} reactions for {chem_name}")
        
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

            valid_i = 0
            mistakes_cnt = 0
            valid_ratio = [0.0 for i in range(len(reactions))]
            while valid_i < valid_cnt:
                response = self.__fetch_llm_answer(messages, self.gpt_oss)
                verdicts = self.__get_verdicts_bool_from_response(response)
                if len(verdicts) != len(reactions):
                    self.log(f"Mistake: {len(verdicts)} != {len(reactions)}: '{response}'")
                    mistakes_cnt += 1
                    if mistakes_cnt > 2:
                        return None
                    continue
                
                for i, verd in enumerate(verdicts):
                    valid_ratio[i] += float(verd)/valid_cnt
                
                valid_i += 1
            
            valid_reacts_cnt = 0
            verdict_reactions = []
            for i, react in enumerate(reactions):
                react['valid'] = valid_ratio[i] > 0.5
                verdict_reactions.append(react)

                if react['valid']:
                    valid_reacts_cnt += 1
            
            self.log(f"{valid_reacts_cnt}/{len(reactions)} are valid. CTT: {self.completion_tokens_total}")
            
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
    

    def __fix_broken_raw_reactions(self, reactions_broken):
        try:
            instruct = \
            "You will be given a list of unbalanced chemical reaction schemes. " \
            "For each scheme, if the reaction can be corrected by replacing the products with valid chemical substances, " \
            "output the corrected unbalanced scheme using chemical names, not formulas. " \
            "Use '->' and '+' markup symbols. " \
            "If the reaction is not possible with the given reagents, output only 'Invalid'. " \
            "Write each result on a new line, matching the order of the input reactions. Do not include any explanations or extra text."

            reactions_str = '\n'.join([f"{i+1}. {react}" for i, react in enumerate(reactions_broken)])

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{instruct}\n{reactions_str}"}
            ]
            for attempt_i in range(2):
                response = self.__fetch_llm_answer(messages, self.gpt_oss)
                response = response.strip('\n').split('\n')
                if len(response) == len(reactions_broken):
                    break
            else:
                self.log(f"Failed to fetch fix")
                return None
            
            reactions_fixed = []

            for i, line in enumerate(response):
                if 'invalid' in line.lower():
                    continue

                reactions_fixed.append({'cid': None, 'reaction': line})

            print(f"Fixed {len(reactions_fixed)}/{len(reactions_broken)} reactions. CTT: {self.completion_tokens_total}")
            
            return reactions_fixed


        except Exception as e:
            self.log(f"Exception in '__fix_broken_raw_reactions': {e}")
            return None


    def fix_broken_raw_reactions(self, max_workers=1):
        with open(self.raw_reactions_verdict_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        processed = self.__get_processed_entries(self.raw_reactions_staged_fn, 'reaction')

        reactions_broken = []

        with open(self.raw_reactions_staged_fn, 'a') as f_out:
            for entry in entries:
                reaction = entry['reaction']
                valid = entry['valid']
                cid = entry['cid']

                if reaction in processed:
                    continue

                if not valid:
                    reactions_broken.append(reaction)
                else:
                    f_out.write(json.dumps({'cid': cid, 'reaction': reaction}) + '\n')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(self.raw_reactions_staged_fn, 'a') as f_out:
            futures = []
            i = 0
            while i < len(reactions_broken):
                futures.append(executor.submit(self.__fix_broken_raw_reactions, reactions_broken[i:i+10]))
                i += 10
            
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                for react in res:
                    f_out.write(json.dumps(react) + '\n')
                    f_out.flush()

            
            


    def deduplicate_raw_reactions(self):
        with open(self.raw_reactions_fn) as f:
            entries = [json.loads(x) for x in f.read().strip().split('\n')]
        
        raw_reacts = dict()
        for entry in entries:
            cid = entry['cid']
            reactions = entry['reactions']
            if cid not in raw_reacts:
                raw_reacts[cid] = []
            raw_reacts[cid] += reactions
        
        with open(self.raw_reactions_fn, 'w') as f:
            for cid in raw_reacts:
                entry = {'cid': cid, 'reactions': raw_reacts[cid]}
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



if __name__ == "__main__":
    chemsllm = ChemsLLM("chems.jsonl", "data/")
    #chemsllm.get_raw_reactions("raw_reactions.jsonl", max_workers=20)
    #chemsllm.process_raw_reactions('raw_reactions.jsonl')
    #chemsllm.deduplicate_raw_reactions('raw_reactions.jsonl')
    chemsllm.validate_raw_reactions(max_workers=10)
    #chemsllm.fix_broken_raw_reactions(max_workers=1)
    
