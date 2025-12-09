import pdfplumber
import os

# Konfigurasi
PDF_DIR = "./data/pdfs"
OUTPUT_FILE = "./data/raw_text.txt"

def extract_text():
    full_text = ""
    # Cek folder
    if not os.path.exists(PDF_DIR):
        print("Folder data/pdfs tidak ditemukan! Buat folder dan isi PDF.")
        return

    print("Mulai mengekstrak teks dari PDF...")
    files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    for pdf_file in files:
        path = os.path.join(PDF_DIR, pdf_file)
        print(f"Memproses: {pdf_file}...")
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
        except Exception as e:
            print(f"Error pada {pdf_file}: {e}")

    # Simpan hasil
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Selesai! Teks tersimpan di {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_text()