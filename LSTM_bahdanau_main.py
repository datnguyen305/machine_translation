# --- IMPORTS ---
from vocab import Vocab
from config.config import BahdanauLSTM_config as Config
from models.LSTM_bahdanau_atttention import BahdanauLSTM
from dataset import PhoMTDataset, collate_fn

# Import từ file metric của bạn (đã đổi tên)
from custom_metric import Bleu, Rouge 

# Import thư viện Hugging Face
import evaluate as hf_evaluate 
import nltk

from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from tqdm import tqdm 
import os 
import time

# --- TẢI DATA ---
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# --- CẤU HÌNH ---
os.makedirs("checkpoints", exist_ok=True)
BEST_MODEL_PATH = "checkpoints/best_model.pt"
path = "/kaggle/input/d/mymomsdatatset/phomt-dataset"

# --- KHỞI TẠO ---
print("Loading vocab ... ")
vocab = Vocab(path=path, src_language="vietnamese", tgt_language="english")

print("Creating model ... ")
config = Config()
model = BahdanauLSTM(vocab, config).to(config.device)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx) 
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Load METEOR
print("Loading METEOR metric...")
meteor_hf = hf_evaluate.load('meteor')

print("Loading dataset ... ")
# (Giữ nguyên phần load dataset của bạn)
train_dataset = PhoMTDataset(path="/kaggle/input/d/mymomsdatatset/phomt-dataset/train.json", vocab=vocab)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)

dev_dataset = PhoMTDataset(path="/kaggle/input/d/mymomsdatatset/phomt-dataset/dev.json", vocab=vocab)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

test_dataset = PhoMTDataset(path="/kaggle/input/d/mymomsdatatset/phomt-dataset/test.json", vocab=vocab)   
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

# --- HÀM ĐÁNH GIÁ (ĐÃ SỬA: TÍNH METEOR TRƯỚC) ---
def evaluate_model(model, dataloader, vocab, device, metrics=None, src_language="vietnamese", tgt_language="english"):
    if metrics is None: metrics = []
        
    model.eval()
    total_loss = 0
    gens = {}  
    gts = {}   
    sample_index = 0 
    
    need_predictions = len(metrics) > 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch[src_language].to(device)
            tgt = batch[tgt_language].to(device)
            
            loss = model(src, tgt) 
            total_loss += loss.item()
            
            if need_predictions:
                prediction_tokens = model.predict(src,tgt) 
                
                prediction_sentences_list = vocab.decode_sentence(prediction_tokens, tgt_language)
                prediction_sentence = prediction_sentences_list[0]
                
                label_sentences_list = vocab.decode_sentence(tgt, tgt_language)
                label_sentence = label_sentences_list[0]

                id = str(sample_index)
                gens[id] = [prediction_sentence] 
                gts[id] = [label_sentence]     
                sample_index += 1
                
    avg_loss = total_loss / len(dataloader)
    metrics_scores = {}
    
    # 3. Tính toán Metrics
    if need_predictions and len(gens) > 0:
        
        # ===> ƯU TIÊN TÍNH METEOR TRƯỚC <===
        if 'meteor' in metrics:
            try:
                # Convert data
                predictions_list = [v[0] for v in gens.values()]
                references_list = [v[0] for v in gts.values()]
                
                # Compute
                result = meteor_hf.compute(predictions=predictions_list, references=references_list)
                val = result['meteor']
                metrics_scores['METEOR'] = (val, val)
                
                # IN NGAY LẬP TỨC ĐỂ KIỂM TRA
                print(f"   >>> [DEBUG] METEOR Calculated: {val*100:.2f}%")
                
            except Exception as e:
                print(f"   >>> [ERROR] METEOR Failed: {e}")
                metrics_scores['METEOR'] = (0.0, 0.0)

        # --- SAU ĐÓ MỚI TÍNH BLEU ---
        if 'bleu' in metrics:
            bleu_metric = Bleu()
            metrics_scores['BLEU'] = bleu_metric.compute_score(gts, gens) 
        
        # --- TÍNH ROUGE ---
        if 'rouge' in metrics:
            rouge_metric = Rouge()
            metrics_scores['ROUGE'] = rouge_metric.compute_score(gts, gens)
            
    return avg_loss, metrics_scores


# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---
print("Starting training ... ")

# THAY ĐỔI 1: Theo dõi BLEU-4 để Early Stopping
best_bleu4_score = -1.0 
patience_limit = 5  
patience_counter = 0

for epoch in range(config.num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch in train_bar:
        src = batch["vietnamese"].to(config.device)
        tgt = batch["english"].to(config.device)

        optimizer.zero_grad()
        loss = model(src, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) 
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    # --- EVALUATION ---
    print(f"Evaluating Epoch {epoch+1}...")
    
    # Tính toán METEOR và BLEU
    dev_loss, dev_metrics = evaluate_model(
        model, 
        dev_dataloader, 
        vocab, 
        config.device, 
        metrics=['meteor', 'bleu'] 
    )
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    # --- LẤY GIÁ TRỊ METRIC ---
    
    # 1. Lấy METEOR (để in ra xem tham khảo)
    current_meteor = 0.0
    if 'METEOR' in dev_metrics:
        current_meteor = dev_metrics['METEOR'][0] if isinstance(dev_metrics['METEOR'], tuple) else dev_metrics['METEOR']

    # 2. Lấy BLEU-4 (Mục tiêu chính để Early Stopping)
    current_bleu4 = 0.0
    if 'BLEU' in dev_metrics:
        # dev_metrics['BLEU'] thường trả về tuple: ([b1, b2, b3, b4], details)
        score_list = dev_metrics['BLEU'][0]
        
        # Kiểm tra xem list có đủ 4 phần tử không
        if isinstance(score_list, list) and len(score_list) >= 4:
            current_bleu4 = score_list[3] # <--- THAY ĐỔI QUAN TRỌNG: Index 3 là BLEU-4
        else:
            print(" [WARNING] Không tìm thấy điểm BLEU-4 trong kết quả trả về.")

    # --- IN KẾT QUẢ ---
    print(f"--- Epoch {epoch+1:02d} Complete (Time: {epoch_mins}m {epoch_secs}s) ---")
    print(f"Training Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
    print(f"⭐️ Dev METEOR: {current_meteor*100:.2f}%")
    
    # In BLEU-4
    print(f"⭐️ Dev BLEU-4: {current_bleu4*100:.2f}%") 

    # --- EARLY STOPPING CHECK (Dựa trên BLEU-4) ---
    if current_bleu4 > best_bleu4_score:
        best_bleu4_score = current_bleu4
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f" >>> SAVED: New best model found (BLEU-4: {best_bleu4_score*100:.2f}%)")
        patience_counter = 0 
    else:
        patience_counter += 1
        print(f" >>> No improvement in BLEU-4. Patience: {patience_counter}/{patience_limit}")

    if patience_counter >= patience_limit:
        print(f"\n*** EARLY STOPPING TRIGGERED (No improve in BLEU-4 for {patience_limit} epochs) ***")
        break
    print("\n")
# --- ĐÁNH GIÁ CUỐI CÙNG ---

print("\nEvaluating on test set using the BEST saved model ... ")

try:
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"Đã tải mô hình tốt nhất từ {BEST_MODEL_PATH}")
    else:
        print("Không tìm thấy file model checkpoint.")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")

test_loss, test_metrics = evaluate_model(
    model, 
    test_dataloader, 
    vocab, 
    config.device, 
    metrics=['meteor', 'bleu', 'rouge'] # Ưu tiên thứ tự list này
)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"Test Loss: {test_loss:.4f}")

# 1. IN METEOR TRƯỚC (Theo yêu cầu của bạn)
if 'METEOR' in test_metrics:
    val = test_metrics['METEOR'][0] if isinstance(test_metrics['METEOR'], tuple) else test_metrics['METEOR']
    print(f"METEOR: {val*100:.2f}%")

# 2. SAU ĐÓ IN BLEU
if 'BLEU' in test_metrics:
    bleu_scores = test_metrics['BLEU'][0]
    if isinstance(bleu_scores, list) and len(bleu_scores) >= 4:
        print(f"BLEU-1: {bleu_scores[0]*100:.2f}%")
        print(f"BLEU-2: {bleu_scores[1]*100:.2f}%")
        print(f"BLEU-3: {bleu_scores[2]*100:.2f}%")
        print(f"BLEU-4: {bleu_scores[3]*100:.2f}%")
    else:
        print(f"BLEU Score: {bleu_scores}")

# 3. CUỐI CÙNG IN ROUGE
if 'ROUGE' in test_metrics:
    val = test_metrics['ROUGE'][0] if isinstance(test_metrics['ROUGE'], tuple) else test_metrics['ROUGE']
    print(f"ROUGE-L: {val*100:.2f}%")