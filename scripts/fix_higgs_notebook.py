import json
import os

filepath = r'c:\Users\diego\OneDrive\Documentos\Cosmologia\A-NumInflation\notebooks\Higgs_USR_Detailed_Scan.ipynb'

if not os.path.exists(filepath):
    # Try relative path if absolute fails (e.g. env diffs)
    filepath = 'notebooks/Higgs_USR_Detailed_Scan.ipynb'

print(f"Processing {filepath}...")
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find first code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        # Check if we need to modify
        has_append = any("sys.path.append" in line for line in source)
        has_insert = any("sys.path.insert" in line for line in source)
        
        if has_insert:
            print("Already has sys.path.insert.")
            break
            
        if has_append:
            print("Found sys.path.append, replacing with sys.path.insert...")
            new_source = []
            for line in source:
                if "sys.path.append(os.path.abspath('..'))" in line:
                    new_source.append("root_dir = os.path.abspath('..')\n")
                    new_source.append("if root_dir not in sys.path:\n")
                    new_source.append("    sys.path.insert(0, root_dir)\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
        else:
            print("No path modification found, prepending...")
            new_lines = [
                "import sys\n",
                "import os\n",
                "root_dir = os.path.abspath('..')\n",
                "if root_dir not in sys.path:\n",
                "    sys.path.insert(0, root_dir)\n",
                "\n"
            ]
            cell['source'] = new_lines + source
            
        break

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Done.")
