import json
import pdfplumber as pdfp
import re


def __parse_page_raw(words, chars, column_headers_fields_map: dict, headers_cutoff_thr_bottom: float, headers_min_top: float, mandatory_field: str, row_min_height):
    column_headers = list(column_headers_fields_map.keys())
    header_to_i = {h: i for i, h in enumerate(column_headers)}
    column_placeholder = 2**30
    column_left_x = [column_placeholder for _ in column_headers]
    for w in words:
        if w['top'] < headers_min_top:
            continue
        if w['bottom'] > headers_cutoff_thr_bottom:
            continue
        text = w['text']

        if text in header_to_i:
            curr_value = column_left_x[header_to_i[text]]
            if curr_value > w['x0']:
                column_left_x[header_to_i[text]] = w['x0']
    
    if any(x == column_placeholder for x in column_left_x):
        print(column_left_x)
        raise Exception("Failed to find all defined column headers")

    tops = sorted(w['top'] for w in words if w['bottom'] > headers_cutoff_thr_bottom)
    rows_top_y = [tops[0]]
    for top in tops[1:]:
        if top - rows_top_y[-1] > row_min_height:
            rows_top_y.append(top)
    
    table = [[[] for c in column_left_x] for r in rows_top_y]

    def binary_search_index(arr, value):
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] < value:
                result = mid  # This could be our answer, but check if there's a later one
                left = mid + 1
            else:
                right = mid - 1
        
        return result

    char_height_thr = 5.0
    for c in chars:
        if c['height'] < char_height_thr:
            continue
        if c['bottom'] < headers_cutoff_thr_bottom:
            continue
        text = c['text']
        row_i = binary_search_index(rows_top_y, c['bottom'])
        col_i = binary_search_index(column_left_x, c['x1'])
        table[row_i][col_i].append(c)
    
    for row_i in range(len(table)):
        for col_i in range(len(table[row_i])):
            table[row_i][col_i].sort(key=lambda c: (round(c['top'], 1), c['x0']))
            table[row_i][col_i] = ''.join([c['text'] for c in table[row_i][col_i]])
    

    def process_entry(entry):
        entry = entry.strip()
        return entry if entry else None
    
    results = []
    for row_i in range(len(table)):
        curr_res = dict()
        for h_i, header in enumerate(column_headers):
            res_field = column_headers_fields_map[header]
            if not res_field:
                continue
            curr_res[res_field] = process_entry(table[row_i][h_i])
        results.append(curr_res)
    

    results_merged = []
    for entry in results:
        if entry[mandatory_field] is not None:
            results_merged.append(entry)
            continue

        if all(entry[field] is None for field in entry.keys()):
            continue

        for field in entry.keys():
            if entry[field] is not None and results_merged[-1][field] is not None:
                # Filtering invisible garbage text
                if '.indb' in entry[field]:
                    continue
                results_merged[-1][field] += f" {entry[field]}"
    

    return results_merged




def parse_organic_constants(crc_fn, out_fn, start_page=1, last_page=-1):
    with pdfp.open(crc_fn) as pdf, open(out_fn, 'w') as f_out:
        page_i = start_page-1
        last_page_i = last_page if last_page != -1 else len(pdf.pages)
        column_headers_fields_map = {"No.": 'ind',
                                    "Name": 'name',
                                    "Synonym": 'synonym',
                                    "Mol.": None,
                                    "CAS": 'cas',
                                    "Wt.": None,
                                    "Physical": 'physical_form',
                                    "mp/\u02daC": 'mp',
                                    "bp/\u02daC": 'bp',
                                    "den/": 'density',
                                    "n": 'refractive_index',
                                    "Solubility": 'solubility'
                                    }
        headers_cutoff_thr_bottom = 80.0
        headers_min_top = 45.0
        row_min_height = 5.0
        for page_i in range(start_page-1, last_page_i, 2):
            page = pdf.pages[page_i]
            words = page.extract_words()
            chars = page.chars

            try:
                page_result = __parse_page_raw(words, chars, column_headers_fields_map, headers_cutoff_thr_bottom, headers_min_top, 'ind', row_min_height)
            except Exception as e:
                print(f"Exception: '{e}'")
                return

            for res in page_result:
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()



def parse_inorganic_constants(crc_fn, out_fn, start_page=1, last_page=-1):
    with pdfp.open(crc_fn) as pdf, open(out_fn, 'w') as f_out:
        start_page_i = start_page-1
        page_i = start_page_i
        last_page_i = last_page if last_page != -1 else len(pdf.pages)
        row_min_height = 5.0
        column_headers_fields_map = {"No.": 'ind',
                                    "Name": 'name',
                                    "Formula": None,
                                    "CAS": 'cas',
                                    "Mol.": None,
                                    "Physical": 'physical_form',
                                    "mp/°C": 'mp',
                                    "bp/°C": 'bp',
                                    "Density": 'density',
                                    "g/100": 'solubility_aq',
                                    "Qualitative": 'solubility'
                                    }
        for page_i in range(start_page-1, last_page_i):
            page = pdf.pages[page_i]
            words = page.extract_words()
            chars = page.chars
            is_first_page = page_i==start_page_i
            headers_cutoff_thr_bottom = 550.0 if is_first_page else 80.0
            headers_min_top = 525 if is_first_page else 60.0

            try:
                page_result = __parse_page_raw(words, chars, column_headers_fields_map, headers_cutoff_thr_bottom, headers_min_top, 'ind', row_min_height)
            except Exception as e:
                print(f"Exception: '{e}'")
                return

            for res in page_result:
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()


def parse_flammability(crc_fn, out_fn, start_page=1, last_page=-1):
    with pdfp.open(crc_fn) as pdf, open(out_fn, 'w') as f_out:
        start_page_i = start_page-1
        page_i = start_page_i
        last_page_i = last_page if last_page != -1 else len(pdf.pages)
        row_min_height = 5.0
        column_headers_fields_map = {"mol.": 'formula',
                                    "name": 'name',
                                    "t": None,
                                    "fp/\u00b0C": 'flash_point',
                                    "fl.": "flash_limits",
                                    "it/\u00b0C": 'ignition_temp'
                                    }
        for page_i in range(start_page-1, last_page_i):
            page = pdf.pages[page_i]
            words = page.extract_words()
            chars = page.chars
            is_first_page = page_i==start_page_i
            headers_cutoff_thr_bottom = 430.0 if is_first_page else 96.0
            headers_min_top = 409 if is_first_page else 85.0

            page_result = __parse_page_raw(words, chars, column_headers_fields_map, headers_cutoff_thr_bottom, headers_min_top, 'formula', row_min_height)
            try:
                pass
            except Exception as e:
                print(f"Exception: '{e}'")
                return

            for res in page_result:
                if not res['name'] or not res['formula']:
                    continue
                res.pop('formula')
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
            


            

parse_organic_constants('data/assets/crc_handbook/organic_constants.pdf', 'data/crc_handbook/organic_constants.jsonl')
parse_inorganic_constants('data/assets/crc_handbook/inorganic_constants.pdf', 'data/crc_handbook/inorganic_constants.jsonl')
parse_flammability('data/assets/crc_handbook/flammability.pdf', 'data/crc_handbook/flammability.jsonl')