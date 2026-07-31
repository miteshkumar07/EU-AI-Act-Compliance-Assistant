import re
import os
import fitz  # PyMuPDF

def clean_page(raw_text):
    text = raw_text
    # Removing unwanted junk
    text = re.sub(r'ELI: http://data\.europa\.eu/eli/reg/\d+/\d+/oj', '', text)
    text = re.sub(r'OJ L,\s*12\.7\.2024(?:,\s*p\.\s*\d+)?(?:\s*EN)?', '', text)
    text = re.sub(r'\b\d+/144\b', '', text)
    text = text.replace("Official Journal\nof the European Union", "")
    text = text.replace("EN\nL series", "")
    
    # Protecting the real paragraph breaks
    text = text.replace('\n\n', '<PARAGRAPH_BREAK>')
    
    # Replacing \n with a space to avoid breaking sentences in the middle
    text = text.replace('\n', ' ')
    # Restoring the paragraph breaks
    text = text.replace('<PARAGRAPH_BREAK>', '\n\n')
    # Sometimes removing headers leaves behind random double spaces. 
    # This regex converts any multiple spaces (2 or more) into a single space.
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# Main processing
doc = fitz.open('./data/raw/Official EUR-Lex.pdf')

# 1. Create an empty string to hold the entire law
full_clean_text = ""

print(f"Processing all {len(doc)} pages... This might take a few seconds.")

# 2. Loop through every single page dynamically
for page_num in range(len(doc)):
    raw = doc[page_num].get_text()
    cleaned = clean_page(raw)
    
    # Add the cleaned page to our master string, plus a double line break 
    # so pages don't crash into each other.
    full_clean_text += cleaned + "\n\n"

# 3. Save the master string to a standard .txt file
output_path = './data/processed/clean_eu_ai_act.txt'

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(full_clean_text)
print(f"Wrote {len(full_clean_text)} bytes to {output_path}")