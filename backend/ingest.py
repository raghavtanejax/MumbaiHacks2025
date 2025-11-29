import os
from rag import ingest_pdf

PDF_DIRECTORY = "./data/pdfs"

def main():
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"Created directory {PDF_DIRECTORY}. Please put your medical PDFs there.")
        return

    print(f"Scanning {PDF_DIRECTORY} for PDFs...")
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_DIRECTORY, filename)
            print(f"Ingesting {filename}...")
            ingest_pdf(file_path, source_name=filename)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
