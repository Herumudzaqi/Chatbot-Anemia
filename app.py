from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

app = Flask(__name__)

# --- CEK GPU DULU ---
if not torch.cuda.is_available():
    print("\n" + "!"*50)
    print("ERROR: GPU NVIDIA TIDAK DITEMUKAN!")
    print("Pastikan Anda sudah install PyTorch versi CUDA (Langkah 1).")
    print("!"*50 + "\n")
    sys.exit(1)

# --- KONFIGURASI MODEL ---
BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ADAPTER_PATH = "./model/lora_adapter"

print("Sedang memuat model ke GPU NVIDIA... (Mohon tunggu)")

try:
    # 1. Konfigurasi Kuantisasi (Agar hemat memori VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 3. Load Base Model (Langsung masuk GPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config, # Pakai config baru
        device_map="auto"
    )
    
    # 4. Gabungkan dengan Adapter Lora
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Model BERHASIL dimuat di GPU!")

except Exception as e:
    print(f"❌ Error Load Model: {e}")
    sys.exit(1)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
Anda adalah asisten kesehatan untuk skrining anemia remaja putri.
Analisis risiko berdasarkan gejala mata (konjungtiva) dan kuku.
JANGAN meminta data pribadi.
Jika gejala berisiko (pucat/lemas), WAJIB sarankan cek Hb Lab dan ke Dokter.
"""

def get_ai_response(user_msg):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    try:
        # Kirim input ke GPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.6,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response_text.split("assistant")[-1].strip()
        
    except Exception as e:
        print(f"Error Generasi: {e}")
        return "Maaf, terjadi kesalahan pemrosesan."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    msg = data.get('message', '')
    if not msg: return jsonify({"reply": "Mohon masukkan pesan."})
    
    reply = get_ai_response(msg)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)