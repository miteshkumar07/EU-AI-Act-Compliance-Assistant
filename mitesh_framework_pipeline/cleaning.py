import pymupdf4llm
import fitz

def clean_pdf_to_markdown(pdf_path, output_path, pages_to_parse):
    """
    Converts the pdf to markdown for chunking by using the pymudf4llm library.
    We are also removing the headers and footers from the markdown to avoid unnecessary repetition in the final output.
    """
    pdf = fitz.open(pdf_path)
    md_converted = pymupdf4llm.to_markdown(pdf, pages=pages_to_parse, header=False, footer=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_converted)
    print(f"Pdf to markdown conversion completed for {pdf_path}")

if __name__ == "__main__":
    pdf_path = ["../data/raw/Official EUR-Lex.pdf", "../data/raw/EU Commission Guidelines on Prohibited AI.pdf"]
    output_path = ["../data/processed/clean_eu_ai_act.md", "../data/processed/clean_eu_proh_ai_act.md"]
    pages_to_parse = [list(range(0, 144)), list(range(5, 134))]
    for pdf in range(len(pdf_path)):
        clean_pdf_to_markdown(pdf_path[pdf], output_path[pdf], pages_to_parse[pdf])