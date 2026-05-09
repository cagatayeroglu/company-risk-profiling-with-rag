import os
import json
import numpy as np
from src.collector import get_cik_for_ticker, get_latest_10k_filing, download_10k_document
from src.extractor import process_single_file
from src.chunker import chunk_single_company
from src.embedder import load_chunks, generate_embeddings, build_faiss_index, save_index
from src.risk_extractor import RiskExtractor
from config import EMBEDDINGS_DIR, CHUNKS_DIR

def run_live_analysis(ticker: str, status_placeholder=None):
    ticker = ticker.upper()
    company_name = f"{ticker} Inc."
    
    def update_status(msg):
        if status_placeholder:
            status_placeholder.info(msg)
        print(msg)
    
    try:
        update_status(f"1/5: SEC'den {ticker} 10-K raporu aranıyor...")
        cik = get_cik_for_ticker(ticker)
        if not cik: return False, f"CIK kodu bulunamadı: {ticker}"
        filing_info = get_latest_10k_filing(cik, ticker)
        if not filing_info: return False, "Son 10-K raporu bulunamadı."
        local_path = download_10k_document(filing_info)
        
        update_status(f"2/5: Risk Faktörleri (Item 1A) çekiliyor...")
        result = process_single_file(local_path, ticker, company_name)
        if not result: return False, "Item 1A metni çıkarılamadı."
        
        update_status(f"3/5: Metin parçalanıyor (Chunking)...")
        text_path = result["output_file"]
        new_chunks = chunk_single_company(ticker, company_name, text_path, filing_info["filing_date"][:4])
        
        update_status(f"4/5: Vektörleştirme ve FAISS indexleme yapılıyor...")
        try:
            old_chunks = load_chunks()
            emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
            if os.path.exists(emb_path):
                old_embeddings = np.load(emb_path)
            else:
                old_embeddings = np.empty((0, 384), dtype=np.float32)
        except Exception:
            old_chunks = []
            old_embeddings = np.empty((0, 384), dtype=np.float32)
            
        new_embeddings = generate_embeddings(new_chunks)
        all_embeddings = np.vstack([old_embeddings, new_embeddings])
        all_chunks = old_chunks + new_chunks
        
        # Save chunks
        os.makedirs(CHUNKS_DIR, exist_ok=True)
        with open(os.path.join(CHUNKS_DIR, "all_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)
            
        index = build_faiss_index(all_embeddings)
        save_index(index, all_chunks, all_embeddings)
        
        update_status(f"5/5: Llama-3 API ile Yapay Zeka Risk Analizi Yapılıyor...")
        extractor = RiskExtractor()
        extractor.load_model()
        extractor.load_retriever()
        
        profile = extractor.extract_company_profile(ticker)
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "risk_profiles")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual
        with open(os.path.join(output_dir, f"{ticker}_risk_profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False)
            
        # Append to all
        combined_path = os.path.join(output_dir, "all_risk_profiles.json")
        if os.path.exists(combined_path):
            with open(combined_path, "r", encoding="utf-8") as f:
                all_profiles = json.load(f)
            all_profiles = [p for p in all_profiles if p["company"] != ticker]
            all_profiles.append(profile)
        else:
            all_profiles = [profile]
            
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, ensure_ascii=False)
            
        update_status(f"✅ Analiz Başarılı!")
        return True, profile
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Hata oluştu: {str(e)}"
