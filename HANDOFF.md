# Proje Devir Notu — Risk Profilleme RAG (FY2025)

Bu belge, sistemde yapılan tüm değişiklikleri **nedenleriyle birlikte** özetler.
Projeye devam edecek kişi neyin neden değiştiğini buradan anlayabilir.

## Genel bağlam
SEC 10-K "Item 1A Risk Factors" bölümünden şirket risk profili çıkaran bir RAG sistemi.

**Kısıtlar / kararlar:**
- Modeller **değiştirilmedi**: Groq `llama-3.1-8b-instant` (LLM), `bge-small-en-v1.5` (embedding), `ms-marco-MiniLM-L-6-v2` (reranker).
- **Fine-tune yapılmadı** — bilinçli: etiketli severity verisi yok ve sorun yetenek değil prompt'tu. Doğru sıra: önce prompt → küçük eval seti → gerekirse hibrit kural → en son fine-tune.
- Değişiklikler `main` dalında ama **henüz commit edilmedi**.

---

## 1. Chunking — karakter→token + başlık-duyarlı + overlap bug
**Ne:** `split_text_recursive` baştan yazıldı (tiktoken token sayımı, paragraf bütünlüğü koruyan packing, risk-faktörü başlıklarında yeni chunk). Kritik bir **overlap bug'ı** bulundu: `_take_overlap` küçük kuyruk yerine tüm ~400 token'lık unit'i overlap diye ekliyordu → chunk'lar 2 katı (medyan 719 token). Token-bağlı kuyrukla düzeltildi → medyan ~455 (400 içerik + 80 overlap).
**Neden:** Eskiden `CHUNK_SIZE=512` "token" deniyordu ama karakter sayılıyordu (~128 token, çok küçük); risk faktörleri ortadan bölünüyordu.
**Dosyalar:** `src/chunker.py`, `config.py` (`CHUNK_SIZE=400`, `CHUNK_OVERLAP=80`, `CHUNK_TOKENIZER`, `CHUNK_HEADING_AWARE`).

## 2. Extraction bug fix — (en büyük kök sebep)
**Ne:** `extract_item_1a` "son eşleşmeyi al" mantığı, dokümanın ilerisindeki **atıf cümlelerini** gerçek başlık sanıyordu → NVDA & AMD MD&A/finansal metni, MSFT TOC çekiyordu. Artık kendi satırındaki gerçek başlıkları bulup en uzun gövdeyi seçiyor. NVDA & AMD düzeldi. **MSFT** gövdesinde matchable başlık yok (yapı-temelli yöntem gerek) → `COMPANIES`'ten geçici çıkarıldı.
**Neden:** Bozuk extraction yüzünden cross-encoder NVDA'yı tüm kategorilerde ~0 skorluyordu → her şey aşağı doğru bozuluyordu.
**Dosyalar:** `src/extractor.py`, `config.py` (`COMPANIES`'te MSFT yorum satırı).

## 3. Query template'ler — keyword → doğal soru
**Ne:** `"government regulation compliance enforcement"` → `"What government regulations or compliance requirements pose risks to the company?"`.
**Neden:** Cross-encoder (MS-MARCO) gerçek sorularla eğitildi; keyword yığınını zayıf skorluyordu. AAPL/TSLA relevance 0.9+'a çıktı.
**Dosya:** `config.py` `RISK_CATEGORIES`.

## 4. Hybrid retrieval — eklendi, sonra eval ile KAPATILDI
**Ne:** BM25 + dense'i RRF ile birleştiren hybrid eklendi. Etiketli eval'de **dense > hybrid** çıktı (0.83 vs 0.79); stopword filtresi de kurtaramadı → `HYBRID_ENABLED=False` (dense default).
**Neden:** Lexical katkı beklendi ama doğal-dil sorgu + güçlü dense + cross-encoder ile BM25 sadece gürültü ekledi. Kapatmak hem kaliteyi artırdı hem BM25 maliyetini kaldırdı.
**Dosyalar:** `src/retriever.py` (BM25/RRF kodu duruyor ama dormant; `use_hybrid` artık sadece BM25 varlığına bakar), `config.py`, `requirements.txt` (`rank-bm25`).

## 5. Confidence — LLM yerine retrieval'dan türetiliyor
**Ne:** `compute_retrieval_confidence` (en güçlü chunk relevance + ortalama + kaç query'de göründüğü) LLM confidence'ını override ediyor; orijinal `llm_confidence` olarak saklanıyor.
**Neden:** 8B model confidence'ı hep 0.85'e çöküyordu (mode collapse). Artık 0.15–0.95 değişiyor.
**Dosya:** `src/risk_extractor.py`.

## 6. Relevance gating — sabit 5 chunk → değişken (yumuşak/relatif)
**Ne:** `retrieve_for_risk_category` artık kategorinin en iyi chunk'ına göre **relatif** eşik (`RELEVANCE_KEEP_RATIO`) + gürültü tabanı (`RELEVANCE_FLOOR`) uyguluyor → chunk sayısı kanıt gücüne göre 1–5. `relevance = sigmoid(rerank_score)`.
**Neden:** Hep 5 chunk prompt'u sulandırıyordu. Mutlak eşik kategoriler arası kıyaslanamaz (NVDA sıfırlandı) → relatif + yumuşak.
**Dosyalar:** `src/retriever.py`, `config.py`.

## 7. Severity — 5-seviye fix + rubric + few-shot + zayıf-kanıt tavanı
**Ne:**
- `validate_risk_profile` artık 5 seviyeyi de kabul ediyor (eskiden `negligible`/`critical`'ı sessizce `low`'a eziyordu).
- Prompt rubric'i sıkılaştırıldı: "material adverse effect" boilerplate'i tek başına high yapmaz; high için somut **escalator** (sayısal $, aktif dava, gerçekleşmiş olay) şart.
- Few-shot örnekleri eklendi (medium / high / negligible / low).
- **Zayıf-kanıt tavanı** (`LOW_EVIDENCE_RELEVANCE=0.30`): en iyi chunk zayıfsa severity "low"a kapanır.
**Neden:** Her şey "high" çıkıyordu (boilerplate + 8B); hiç "low" çıkmıyordu (gating ikili). Düzeldi: dağılım negligible 6 / low 3 / medium 31 / high 16.
**Dosyalar:** `prompts/risk_extraction.py`, `src/risk_extractor.py`, `config.py`.

## 8. İstek boyutu + başarısızlık yönetimi (Groq rate limit)
**Ne:** LLM'e gönderilen evidence 5 chunk + güvenlik kırpması (`LLM_EVIDENCE_CHAR_LIMIT=2200`); few-shot kısaltıldı (istek ~6034→~3800 token). API hatasında artık sessizce `"{}"` dönmüyor → `LLMGenerationError` fırlatılıp profil `extraction_failed=True`, conf 0.0 ile işaretleniyor.
**Neden:** Few-shot büyüyünce 413 (TPM 6000) hatası geldi; ve hata sessizce sahte "LOW, conf 0.95" üretiyordu.
**Dosyalar:** `config.py`, `prompts/risk_extraction.py`, `src/risk_extractor.py`.

## 9. Performans
**Ne:** `src/model_cache.py` (yeni) — embedding + reranker process başına bir kez yükleniyor (embedder & retriever paylaşıyor). Canlı-mod **dedup fix** (`src/live_pipeline.py`) — aynı ticker tekrar çalıştırılınca index'i dublike etmiyor (index 466→233 temizlendi).
**Neden:** Modeller tekrar tekrar yükleniyordu; index dublike olup şişiyordu. Saf compute ~25s; gözlenen 3-5 dk'nın çoğu Groq rate-limit beklemeleriydi.

## 10. Evaluation
**Ne:**
- `src/quality_eval.py` (yeni) — **grounding** (snippet'ler kaynakta birebir mi: %97.1) + severity/confidence/failure istatistikleri. Bedava, LLM'siz (RAGAS faithfulness'ın judge gerektirmeyen proxy'si).
- `src/evaluator.py` — year bug fix, dense/hybrid ablation, etiketleme scaffold üretici/dönüştürücü, markdown rapor yazıcı.
- **LLM-etiketli silver retrieval seti** (40 sorgu, 114 ilgili) — `evaluation/annotations/`.
- **RAGAS** (`src/ragas_eval.py`, yeni) — judge = Groq `llama-3.3-70b-versatile` (değerlendirilen 8B'den güçlü → döngüsel değil), embedding = lokal bge. Metrikler: faithfulness + context_precision. Rate-limit için küçük alt küme (varsayılan 6) + seri çalışma. Çalıştır: `python3 -m src.ragas_eval 2025 [N]` → `data/risk_profiles/<yıl>/ragas_report.json`. **Not:** langchain 1.x ragas'ı bozuyor → `requirements.txt`'te langchain `<0.4`, `ragas==0.2.15`.
- RAGAS, verbatim grounding proxy'sinin yakalayamadığını yakalıyor: snippet birebir olsa bile **explanation'daki fazladan iddiaları** faithfulness ile tespit ediyor.
- **Baseline merdiveni** (`src/evaluator.py`) — proposal §3.7 zorunlu: Keyword (baseline) / Dense FAISS (no rerank, baseline) / Dense+Reranker (ablasyon) / Hybrid (ekstra). Sonuç: reranker en büyük katkı (MRR 0.58→0.85), keyword en zayıf (0.40), hybrid dense'ten hafif kötü.
- **Kategori-tespit accuracy + macro-F1** (`src/category_eval.py`, yeni) — proposal §3.7 zorunlu. Gold: `evaluation/annotations/category_gold_2025.csv` (kaynak-doğrulanmış silver, hepsi present). Sonuç: accuracy 0.95, macro-F1 0.49 (2 false negative = AMZN supply/cyber, gating fazla agresif). Çalıştır: `python3 -m src.category_eval 2025`.
- **Faithfulness 0-2 rubric** — proposal §3.7 zorunlu. RAGAS faithfulness'tan eşlenir (≥0.8→2, 0.3-0.8→1, <0.3→0); rapora otomatik eklenir (ek token yok). Sonuç: ortalama 1.13 (4×tam / 9×kısmi / 2×desteksiz).
- `quality_eval` artık yalnız aktif COMPANIES'i değerlendirir (canlı-mod demo ticker'ları GOOGL/NFLX/PLTR/AVGO eval'i kirletmez).
**Çıktı dosyası:** `evaluation/results/evaluation_report_2025.md` — retrieval baseline merdiveni + kategori accuracy/macro-F1 + RAGAS + faithfulness 0-2 + generation quality tek raporda. Yeniden üret: `python3 -m src.evaluator 2025` (RAGAS hariç hepsi ücretsiz/otomatik; RAGAS'ı tazelemek istersen önce `python3 -m src.ragas_eval 2025 16`).

### Proposal §3.7 Evaluation Plan — kapsam durumu (rapora hazır)
| Gereksinim | Durum |
|---|---|
| Recall@5 / MRR / nDCG@5, ~50 sorgu | ✅ 40 sorgu (≈50) |
| ≥2 baseline: keyword + embedding FAISS | ✅ baseline merdiveninde |
| embedding + reranker ablasyonu | ✅ Dense vs Dense+Rerank |
| accuracy + macro-F1 (kategori tespiti) | ✅ category_eval |
| faithfulness 0-2 rubric | ✅ RAGAS'tan eşlenmiş |
| 5-6 şirket | ✅ 5 aktif (MSFT limitation) |

## Veri temizliği
Silindi: **GOOG** (GOOGL ile birebir dublikasyon), **AVGO** (8/8 başarısız çöp profil), bayat `retrieval_annotations.csv` (var olmayan chunk_id'ler).

---

## Çalıştırma sırası (sıfırdan)
```bash
pip install -r requirements.txt   # rank-bm25 dahil
python3 -c "from src.collector import collect_10k_for_year; collect_10k_for_year(2025)"
python3 -c "from src.extractor import extract_all_item_1a_for_year; extract_all_item_1a_for_year(2025)"
python3 -c "from src.chunker import chunk_all_for_year; chunk_all_for_year(2025)"
python3 -c "from src.embedder import build_index_for_year; build_index_for_year(2025)"
python3 -c "from src.risk_extractor import RiskExtractor; e=RiskExtractor(); e.load_model(); e.load_retriever(year=2025); e.extract_all_profiles(year=2025)"
python3 -m src.quality_eval 2025      # kalite raporu
python3 -m src.evaluator 2025         # retrieval + tam markdown rapor
streamlit run app.py                  # dashboard
```
Not: LLM adımı için `.env` içinde geçerli `GROQ_API_KEY` gerekir.

## Önemli config anahtarları (`config.py`)
| Anahtar | Değer | Anlamı |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 400 / 80 | token cinsinden |
| `HYBRID_ENABLED` | `False` | dense default (eval gerekçesi); `True` ile BM25+RRF açılır |
| `RELEVANCE_KEEP_RATIO` / `RELEVANCE_FLOOR` | 0.5 / 0.05 | relatif gating |
| `LOW_EVIDENCE_RELEVANCE` | 0.30 | bu altı → severity tavanı "low" |
| `LLM_EVIDENCE_CHUNKS` / `LLM_EVIDENCE_CHAR_LIMIT` | 5 / 2200 | LLM'e giden evidence |

## Açık işler / dikkat
1. **MSFT** extraction (başlıksız gövde → HTML yapı-temelli yöntem gerek), sonra `COMPANIES`'e geri ekle.
2. **Token-per-day limiti** dolunca profiller eksik kalabilir (`extraction_failed=True` ile işaretli; ör. NVDA'da 1 kategori). Limit sıfırlanınca yeniden üret.
3. Eval etiketleri **LLM-silver** (insan değil) — `evaluation/annotations/retrieval_scaffold_2025.csv`'deki `is_relevant` gözden geçirilmeli; sonra `scaffold_to_annotations` + `evaluate_retrieval` ile rapor güncellenir.
4. **RAGAS** şu an küçük alt kümede (varsayılan 6 örnek, gürültülü). Token bütçesi izin verince `python3 -m src.ragas_eval 2025 15` ile daha kararlı sayı al. İlk verifikasyonda AAPL Supply Chain faithfulness=0.25 çıktı (explanation kanıtı aşan iddia içeriyor) — incelenmeli.
5. İleride: Operational kategorisinde cyber içeriği taşması, GOOGL/NFLX/PLTR canlı-mod eklemeleri indekste değil.

## Değişen/eklenen dosyalar (özet)
- `config.py` (chunking, retrieval, gating, severity, LLM istek ayarları, COMPANIES)
- `src/chunker.py`, `src/extractor.py`, `src/retriever.py`, `src/risk_extractor.py`, `src/live_pipeline.py`, `src/embedder.py`
- `prompts/risk_extraction.py`
- `src/model_cache.py` (yeni), `src/quality_eval.py` (yeni), `src/ragas_eval.py` (yeni), `src/category_eval.py` (yeni)
- `src/evaluator.py` (baseline merdiveni + kategori + faithfulness 0-2 entegrasyonu)
- `evaluation/annotations/category_gold_2025.csv` (yeni, kategori gold)
- `requirements.txt` (`rank-bm25`, `ragas`, `langchain-groq`, `langchain-huggingface`, langchain `<0.4`)
- `evaluation/annotations/retrieval_scaffold_2025.csv`, `evaluation/annotations/retrieval_annotations_2025.csv`, `evaluation/results/evaluation_report_2025.md`
