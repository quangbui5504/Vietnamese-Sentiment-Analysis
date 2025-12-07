import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_abbreviations(path="abbreviation.csv"):
    mapping = {}
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (row.get("original") or "").strip().lower()
                v = (row.get("replacement") or "").strip().lower()
                if k and v:
                    mapping[k] = v
    except:
        print("abbreviation.csv not found, using basic preprocessing")
    return mapping

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).strip().lower()
    
    # Load abbreviations
    abbr_dict = load_abbreviations()
    
    # Thay thế từ viết tắt
    for abbr, full in abbr_dict.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text)
    
    # Loại bỏ ký tự đặc biệt bị thừa
    text = re.sub(r'[^\w\s\u00C0-\u1EF9.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_prepare_data():
    try:
        df = pd.read_csv('sentiment_data.csv')
        print(f"✅ Loaded {len(df)} samples from sentiment_data.csv")
    except Exception as e:
        print(f"❌ Error loading sentiment_data.csv: {e}")
        return None, None, None, None
        
    # Loại bỏ dữ liệu null
    df = df.dropna(subset=['text', 'label'])
    
    # chuẩn hóa label về in hoa
    df['label'] = df['label'].str.strip().str.upper()
    
    valid_labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    df = df[df['label'].isin(valid_labels)]
    
    # Mapping
    label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    df['label_id'] = df['label'].map(label_map)
    
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    
    print("🔄 Preprocessing texts...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Loại bỏ text rỗng
    df = df[df['processed_text'].str.len() > 3]
    
    # Giảm dataset size
    if device.type == 'cpu' and len(df) > 4000:
        df = df.sample(n=4000, random_state=42).reset_index(drop=True)
    
    print(f"✅ Final dataset size: {len(df)} samples")
    
    # chia dataset thành 2 tập train/val với tỉ lệ: 80/20
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    
    return train_df, val_df, label_map

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def finetune_phobert():    
    # Load dữ liệu
    train_df, val_df, label_map = load_and_prepare_data()
    if train_df is None:
        return
    
    # Load PhoBERT tokenizer và model
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    
    print("🔧 Creating datasets...")
    train_dataset = SentimentDataset(
        train_df['processed_text'], 
        train_df['label_id'], 
        tokenizer,
        max_length=256
    )
    val_dataset = SentimentDataset(
        val_df['processe        output_dir='./phobert-sentiment-finetuned',
        num_train_epochs=3,
        per_device_train_batch_size=8 if device.type == 'cpu' else 16,
        per_device_eval_batch_size=16 if device.type == 'cpu' else 32,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to=None,
    )
 metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_total_limit=1,
    dataloader_num_workers=0,
    remove_unused_columns=True,
    report_to=None,
)
port_to=None,
)

    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("🔥 Starting training...")
    print("⏰ This may take 20-40 minutes...")
    trainer.train()
    
    # Evaluate
    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final validation accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Save model
    print("💾 Saving fine-tuned model...")
    trainer.save_model('./phobert-sentiment-final')
    tokenizer.save_pretrained('./phobert-sentiment-final')
    
    # Save label map
    import json
    with open('./phobert-sentiment-final/label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print("✅ Fine-tuning completed!")
    
    # Test thử model sau khi train
    print("\n🧪 Quick test:")
    model.eval()
    id_to_label = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    
    test_texts = ["Hôm nay tôi rất vui", "Dở quá", "Bình thường thôi"]
    
    with torch.no_grad():
        for text in test_texts:
            processed = preprocess_text(text)
            inputs = tokenizer(processed, return_tensors='pt', truncation=True, padding=True, max_length=256)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            predicted_label = id_to_label[predicted_class]
            print(f"'{text}' -> {predicted_label} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    finetune_phobert()