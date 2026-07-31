import re
import os
import fitz  # PyMuPDF

def clean_page(raw_text):
    # 1. Split into lines while it's still raw
    lines = raw_text.split('\n')
    
    # Remove the first line if it's just a standalone page number
    if lines and lines[0].strip().isdigit():
        lines[0] = ""
        
    # Remove the last line if it's just a standalone page number
    if lines and lines[-1].strip().isdigit():
        lines[-1] = ""
        
    # Reconstruct the text block
    text = '\n'.join(lines)
    
    # 2. Protect the real paragraph breaks
    text = text.replace('\n\n', '<PARAGRAPH_BREAK>')
    
    # 3. Heal the broken mid-sentence lines
    text = text.replace('\n', ' ')
    
    # 4. Restore the paragraph breaks
    text = text.replace('<PARAGRAPH_BREAK>', '\n\n')
    
    # 5. Final cleanup of double spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# Main processing
doc = fitz.open('./data/raw/EU Commission Guidelines on Prohibited AI.pdf')

# 1. Create an empty string to hold the entire law
full_clean_text = ""

print(f"Processing all {len(doc)} pages... This might take a few seconds.")

# 2. Loop through every single page dynamically
for page_num in range(5, len(doc)):
    raw = doc[page_num].get_text()
    cleaned = clean_page(raw)
    
    # Add the cleaned page to our master string, plus a double line break 
    # so pages don't crash into each other.
    full_clean_text += cleaned + "\n\n"

# 3. Save the master string to a standard .txt file
output_path = './data/processed/clean_eu_prohibited_ai.txt'

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(full_clean_text)
print(f"Wrote {len(full_clean_text)} bytes to {output_path}")