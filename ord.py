import json
import os

from ord_schema.message_helpers import load_message, write_message
from ord_schema.proto import dataset_pb2
from google.protobuf.json_format import MessageToJson

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors

from random import sample


# Disable all RDKit warnings and info messages
RDLogger.DisableLog('rdApp.*')


class OrdParse:
    def __init__(self, ord_data, chems_data):
        self.ord_data = ord_data
        self.chems_data = chems_data
    

    def extract_file(self, in_fn, out_fn):
        dataset = load_message(
            os.path.join(self.ord_data, in_fn),
            dataset_pb2.Dataset,
        )

        # take one reaction message from the dataset for example
        rxn = dataset.reactions[0]
        rxn_json = json.loads(
            MessageToJson(
                message=rxn,
                including_default_value_fields=False,
                preserving_proto_field_name=True,
                indent=2,
                sort_keys=False,
                use_integers_for_enums=False,
                descriptor_pool=None,
                float_precision=None,
                ensure_ascii=True,
            )
        )

        with open(out_fn, 'w') as f:
            f.write(json.dumps(rxn_json, indent=2))
    

    def extract_all_simple(self, out_fn):
        cts = []
        reactions_written = 0
        overall = 0
        for dirpath, dirnames, filenames in os.walk(self.ord_data):
            for fn in filenames:
                file_path = os.path.join(dirpath, fn)
                if file_path.endswith(".pb.gz"):
                    try:
                        dataset = load_message(
                            file_path,
                            dataset_pb2.Dataset,
                        )
                        for reaction in dataset.reactions:
                            bad = False
                            reaction_json = json.loads(
                                MessageToJson(
                                    message=reaction,
                                    including_default_value_fields=False,
                                    preserving_proto_field_name=True,
                                    indent=2,
                                    sort_keys=False,
                                    use_integers_for_enums=False,
                                    descriptor_pool=None,
                                    float_precision=None,
                                    ensure_ascii=True,
                                )
                            )
                            inputs_dict = reaction_json["inputs"]
                            inputs_keys = list(inputs_dict.keys())
                            
                            inputs = []
                            for key in inputs_keys:
                                for input in inputs_dict[key]["components"]:
                                    if input['reaction_role'] == "REACTANT":
                                        input = list(filter(lambda x: x['type'] == "SMILES", input['identifiers']))
                                        if not len(input):
                                            bad = True
                                            break

                                        inputs.append(input[0]['value'])
                            
                            if bad:
                                continue
                            
                            outcomes_list = reaction_json["outcomes"]
                            if len(outcomes_list) > 1:
                                print(file_path)
                                continue
                            
                            outcomes = []
                            for prod in outcomes_list[0]["products"]:
                                prod_smiles = list(filter(lambda x: x['type'] == "SMILES", prod['identifiers']))
                                if not prod_smiles:
                                    bad = True
                                    break
                                outcomes.append(prod_smiles[0]['value'])
                            
                            if bad:
                                continue

                            chems = inputs + outcomes
                            complexities = []
                            for chem in chems:
                                mol = Chem.MolFromSmiles(chem)
                                if not mol:
                                    bad = True
                                    break
                                bertz_ct = GraphDescriptors.BertzCT(mol)
                                complexities.append(bertz_ct)
                            
                            if bad:
                                continue
                            
                            max_ct = max(complexities)
                            overall += 1
                            if max_ct < 500:
                                with open(out_fn, 'a') as f:
                                    f.write(json.dumps(reaction_json) + '\n')
                                reactions_written += 1
                                if reactions_written % 1000 == 0:
                                    print(f"Written: {reactions_written}; All: {overall}")

                    except Exception as e:
                        print(f"Exception occured: {e}")
    

    def split_ord_file(self, ord_file, out_dir, prefix="ord", lines_per_file=20000):
        with open(ord_file) as f_in:
            i = 1
            while True:
                f_index = i // lines_per_file
                out_fn = os.path.join(out_dir, f"{prefix}_{f_index}.jsonl")
                with open(out_fn, 'w') as f_out:
                    print(f"Writing in '{out_fn}'")
                    while (i % lines_per_file) != 0:
                        line = f_in.readline()
                        if not line:
                            return

                        f_out.write(line)
                        i += 1
                    i += 1
    
    def sample_ord(self, ord_file, samples_num, out_fn):
        with open(ord_file) as f:
            reactions = [json.loads(x) for x in f.read().strip().split('\n')]
        
        res = sample(reactions, samples_num)

        with open(out_fn, 'w') as f:
            f.write(json.dumps(res, indent=1))
    

    def clean_ord_reactions(self, ord_files_dir, out_fn):
        fixed_cnt = 0
        with open(out_fn, 'w') as f_out:
            for dirpath, dirnames, filenames in os.walk(ord_files_dir):
                for fn in filenames:
                    with open(os.path.join(dirpath, fn)) as f:
                        for line in f:
                            bad = False

                            reaction = json.loads(line)
                            inputs_dict = reaction["inputs"]
                            inputs_keys = list(inputs_dict.keys())
                            reactants = []
                            solvents = []
                            catalysts = []
                            for key in inputs_keys:
                                for input in inputs_dict[key]["components"]:
                                    input_smiles = list(filter(lambda x: x['type'] == "SMILES", input['identifiers']))
                                    if not input_smiles:
                                        bad = True
                                        break

                                    input_name = list(filter(lambda x: x['type'] == "NAME", input['identifiers']))
                                    entry = {'smiles': input_smiles[0]['value']}
                                    if input_name:
                                        entry['name'] = input_name[0]['value']
                                    else:
                                        entry['name'] = None
                                    if input['reaction_role'] == "REACTANT":
                                        reactants.append(entry)
                                    elif input['reaction_role'] == "SOLVENT":
                                        solvents.append(entry)
                                    elif input['reaction_role'] == "CATALYST":
                                        catalysts.append(entry)
                            
                            if bad:
                                continue
                            
                            products_raw = reaction["outcomes"][0]["products"]
                            products = []
                            for prod in products_raw:
                                prod_smiles = list(filter(lambda x: x['type'] == "SMILES", prod['identifiers']))[0]['value']
                                prod_name = list(filter(lambda x: x['type'] == "NAME", prod['identifiers']))

                                entry = {'smiles': prod_smiles}
                                if prod_name:
                                    entry['name'] = prod_name[0]['value']
                                else:
                                    entry['name'] = None
                                
                                products.append(entry)
                            
                            def fix_chems(chems):
                                nonlocal fixed_cnt
                                fixes = {
                                    "[Li][CH2]CCC": "[Li+].CCC[CH2-]",
                                    "[H][H]": "[HH]",
                                    "[Al+3].[H-].[H-].[H-].[H-].[Li+]": "[Li+].[AlH4-]",
                                    "[Cu][I]": "[Cu+].[I-]",
                                    "[Al+3].[Cl-].[Cl-].[Cl-]": "[Al](Cl)(Cl)Cl",
                                    "O.[Cl-].[Na+]": "[O-]Cl.[Na+]",
                                    "[Mn+4].[O-2].[O-2]": "O=[Mn]=O",
                                    "CC(C)[CH2][Al+][CH2]C(C)C.[H-]": "[H-].CC(C)C[Al+]CC(C)C",
                                    "N#[C][Cu]": "[C-]#N.[Cu+]",
                                    "N#[C][Na]": "[C-]#N.[Na+]",
                                    "[C].[Pd]": "C.[Pd]"
                                }

                                fixed = []
                                negative = []
                                positive = []
                                for chem in chems:
                                    smiles = chem['smiles']
                                    mol = Chem.MolFromSmiles(smiles)
                                    if not mol:
                                        return None
                                    smiles = Chem.MolToSmiles(mol)
                                    if smiles in fixes:
                                        smiles = fixes[smiles]
                                    chem['smiles'] = smiles
                                    charge = Chem.GetFormalCharge(mol)
                                    if charge == 0:
                                        fixed.append(chem)
                                    elif charge > 0:
                                        positive.append(chem)
                                    else:
                                        negative.append(chem)
                                
                                if negative or positive:
                                    if len(negative) == 1 and len(positive) == 1:
                                        smiles = f"{positive[0]['smiles']}.{negative[0]['smiles']}"
                                        mol = Chem.MolFromSmiles(smiles)
                                        if not mol:
                                            return None
                                        if Chem.GetFormalCharge(mol) != 0:
                                            return None
                                        smiles = Chem.MolToSmiles(mol)
                                        if smiles in fixes:
                                            smiles = fixes[smiles]
                                        
                                        fixed_cnt += 1
                                        chem = {'smiles': smiles}
                                        fixed.append(chem)
                                
                                return fixed
                            
                            reactants = fix_chems(reactants)
                            if not reactants:
                                continue
                            products = fix_chems(products)
                            if not products:
                                continue
                            
                            description = None
                            if "notes" in reaction:
                                notes = reaction['notes']
                                if "procedure_details" in notes:
                                    description = notes['procedure_details']
                            
                            provenance = dict()
                            if "provenance" in reaction:
                                provenance_field = reaction['provenance']
                                if 'doi' in provenance_field:
                                    provenance['doi'] = provenance_field['doi']
                                else:
                                    provenance['doi'] = None

                                if 'patent' in provenance_field:
                                    provenance['patent'] = provenance_field['patent']
                                else:
                                    provenance['patent'] = None

                            cleaned_entry = {
                                "reagents": reactants,
                                "products": products,
                                "solvents": solvents if solvents else None,
                                "catalysts": catalysts if catalysts else None,
                                "description": description,
                                "provenance": provenance if provenance else None
                            }

                            f_out.write(json.dumps(cleaned_entry) + '\n')

        print(f"Total fixed: {fixed_cnt}")

                                
        
            





                        
                        


                        


if __name__ == "__main__":
    ord = OrdParse("ord-data/data", "data/")
    #ord.extract_file("d6/ord_dataset-d6cdba90760a47779a36ece5962905eb.pb.gz", "out.json")
    #ord.extract_all_simple("out1.jsonl")
    #ord.split_ord_file("out.jsonl", "ord/")
    #ord.sample_ord("ord/ord_0.jsonl", 5, "samples.json")
    ord.clean_ord_reactions('ord', 'cleaned_ord.jsonl')
