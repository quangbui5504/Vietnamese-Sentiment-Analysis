import os, re, csv, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

class SentimentService:
    def __init__(self, use_tokenize: bool = True, abbr_path: str = "abbreviation.csv"):
        print("🔧 Loading fine-tuned PhoBERT model...")
        self.abbr = self.load_abbreviations(abbr_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load fine-tuned model
        self.load_finetuned_model()

    def load_abbreviations(self, path: str) -> Dict[str, str]:
        mapping = {}
        if not os.path.exists(path):
            print("[⚠] abbreviation.csv not found")
            return mapping
        
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (row.get("original") or "").strip().lower()
                v = (row.get("replacement") or "").strip().lower()
                if k and v:
                    mapping[k] = v
        return mapping

    def load_finetuned_model(self):
        """Load fine-tuned PhoBERT model"""
        model_path = './phobert-sentiment-final'
        
        if not os.path.exists(model_path):
            print(f"❌ Fine-tuned model not found at {model_path}")
            print("🔧 Please run finetune_phobert.py first!")
            raise Exception("Fine-tuned model not found")
        
        try:
            # Load tokenizer và model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            with open(f'{model_path}/label_map.json', 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            
            # Tạo reverse mapping (id -> label)
            self.id_to_label = {v: k for k, v in label_map.items()}
            
            print("✅ Fine-tuned PhoBERT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            raise e

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ""

        text = text.lower().strip()

        # Thay thế từ viết tắt
        for abbr, full in self.abbr.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full, text)

        # Loại bỏ ký tự đặc biệt thừa
        text = re.sub(r'[^\w\s\u00C0-\u1EF9.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze(self, text: str) -> dict:
        """Phân tích sentiment bằng fine-tuned PhoBERT"""
        if not text or len(text.strip()) < 3:
            return {"text": text, "sentiment": "INVALID"}
        
        processed_text = self.preprocess_text(text)
        
        if not processed_text or len(processed_text) < 3:
            return {"text": text, "sentiment": "INVALID"}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=256
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            predicted_label = self.id_to_label[predicted_class]
            
            # Nếu confidence < 0.5, trả về NEUTRAL
            if confidence < 0.5:
                predicted_label = "NEUTRAL"
            
            return {
                "text": processed_text,
                "sentiment": predicted_label,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"[⚠] Fine-tuned model prediction failed: {e}")
            return {"text": processed_text, "sentiment": "NEUTRAL", "confidence": 0.0}
