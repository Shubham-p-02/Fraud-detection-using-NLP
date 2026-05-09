import os
import re

directory = "/Users/shubhampatthe/Downloads/Data/transcripts"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        company_name = filename[:-4]
        company_dir = os.path.join(directory, company_name)
        
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = content.split("____________________________________________________________")
        
        # Don't create directory if there are no transcripts (just to be safe)
        created_dir = False
        
        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not lines:
                continue
                
            title = lines[0]
            if "Table of contents" in title:
                continue
                
            # Try to match Q# YYYY
            match = re.search(r'Q([1-4])\s+(\d{4})', title)
            if match:
                quarter = match.group(1)
                year = match.group(2)
                
                if not created_dir:
                    os.makedirs(company_dir, exist_ok=True)
                    created_dir = True
                
                base_name = f"Q_{quarter}_ {year}"
                out_filename = f"{base_name}.txt"
                out_filepath = os.path.join(company_dir, out_filename)
                
                # Handle duplicates
                counter = 1
                while os.path.exists(out_filepath):
                    out_filename = f"{base_name}_{counter}.txt"
                    out_filepath = os.path.join(company_dir, out_filename)
                    counter += 1
                    
                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    out_f.write(block.strip() + "\n")
                    
        # Optional: could remove the original file if requested, but better to keep it unless asked.
