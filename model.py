import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TimesformerModel, TrainingArguments, Trainer
)
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from peft import LoraConfig, get_peft_model
import json
import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8
LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"

class SportVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Freeze vision encoder
        self.vision_encoder = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k600",
            num_frames = NUM_FRAMES,
        )
        
        self.vision_encoder.requires_grad_(False)
        
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_CHECKPOINT, torch_dtype=torch.float16)
        
        # Video projector
        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, self.llm.config.hidden_size),
            torch.nn.Dropout(0.1)
        )
        
        # Freeze LLM and add LoRA
        self.llm.requires_grad_(False)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.print_trainable_parameters()
        
    def print_trainable_parameters(self):
        trainable = 0
        total = 0
        for _, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"Trainable: {trainable:,} | Total: {total:,} | %: {100*trainable/total:.2f}")

    def build_prompt(self):
        return """<s>[INST] <<SYS>>
        Analyze this sports video. Consider:
        - Movement quality
        - Technical execution
        - Overall proficiency
        <</SYS>>\n\n[/INST]"""

    def forward(self, video, input_ids, attention_mask, labels):
        # Video features
        video_features = self.vision_encoder(video).last_hidden_state.mean(dim=1)
        video_emb = self.video_adapter(video_features).unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Combine modalities
        inputs_embeds = torch.cat([video_emb.to(text_emb.dtype), text_emb], dim=1)  # [batch_size, 1 + seq_len, hidden_size]
        
        # Create combined attention mask: prepend a mask for the video token
        combined_attention_mask = torch.cat([
            torch.ones(video.size(0), 1, device=DEVICE, dtype=attention_mask.dtype),  # [batch_size, 1]
            attention_mask  # [batch_size, seq_len]
        ], dim=1)  # [batch_size, 1 + seq_len]
        
        # Adjust labels to match the combined input length
        # Prepend -100 (ignore index) for the video token
        adjusted_labels = torch.cat([
            torch.full((labels.size(0), 1), -100, device=DEVICE, dtype=labels.dtype),  # [batch_size, 1]
            labels  # [batch_size, seq_len]
        ], dim=1)  # [batch_size, 1 + seq_len]
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=adjusted_labels
        )
        return outputs

class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, tokenizer):
        self.tokenizer = tokenizer
        self.transform = Compose([
            Resize((224, 224)),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and validate annotations
        with open(annotation_path) as f:
            self.annotations = []
            for idx, line in enumerate(f):
                try:
                    ann = json.loads(line)
                    assert "video_path" in ann, f"Missing video_path in line {idx}"
                    assert "analysis" in ann, f"Missing analysis in line {idx}"
                    assert "proficiency_level" in ann, f"Missing proficiency_level in line {idx}"
                    self.annotations.append(ann)
                except Exception as e:
                    print(f"Skipping invalid annotation {idx}: {str(e)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            ann = self.annotations[idx]
            
            # Load video with fallback
            video = torch.zeros((1, NUM_FRAMES, 3, 224, 224), device=DEVICE)
            if os.path.exists(ann["video_path"]):
                try:
                    video = load_video(ann["video_path"])
                except Exception as e:
                    print(f"Error loading {ann['video_path']}: {str(e)}")
            
            # Ensure video tensor has correct shape
            if video.dim() != 5 or video.size(2) != 3:
                raise ValueError(f"Invalid video tensor shape: {video.shape}")
            
            # Format text
            text = f"""<s>[INST] <<SYS>>
            Analyze this climbing video.
            Classify proficiency as: Novice, Early Expert, Intermediate Expert, Late Expert.
            <</SYS>>\n\n[/INST]
            Analysis: {ann["analysis"]}
            Proficiency: {ann["proficiency_level"]}</s>"""
            
            # Tokenize with error handling
            tokenized = self.tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                'video': video.squeeze(0),  # Remove batch dimension for dataset
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0),
                'labels': tokenized['input_ids'].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return {
                'video': torch.zeros((NUM_FRAMES, 3, 224, 224), device=DEVICE),
                'input_ids': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.zeros(512, dtype=torch.long),
                'labels': torch.zeros(512, dtype=torch.long)
            }

def train():
    model = SportVLM().to(DEVICE)
    
    # Initialize dataset with validation
    train_dataset = VideoTextDataset(
        annotation_path="/data/users/edbianchi/ProfiVLM/annotations/annotation.jsonl",
        tokenizer=model.tokenizer
    )
    
    # Verify dataset
    if len(train_dataset) == 0:
        raise ValueError("No valid training samples found!")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduce batch size to lower memory usage
        learning_rate=1e-3,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,  # Reduce gradient accumulation steps
        fp16=torch.cuda.is_available(),  # Enable mixed precision
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=0  # Disable multiprocessing for data loading
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: {
            'video': torch.stack([x['video'] for x in batch]).to(DEVICE),
            'input_ids': torch.stack([x['input_ids'] for x in batch]).to(DEVICE),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]).to(DEVICE),
            'labels': torch.stack([x['labels'] for x in batch]).to(DEVICE)
        }
    )
    trainer.train()
    trainer.save_model("trained_model")

def generate_analysis(model, video_path):
    model.eval()
    try:
        video = load_video(video_path)
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        return "Video analysis failed: invalid input"
    
    with torch.no_grad():
        video_features = model.vision_encoder(video).last_hidden_state.mean(dim=1)
        video_emb = model.video_adapter(video_features)
        
        prompt = model.build_prompt()
        tokenized = model.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        text_emb = model.llm.get_input_embeddings()(tokenized.input_ids)
        
        inputs_embeds = torch.cat([
            video_emb.unsqueeze(1).to(model.llm.dtype),
            text_emb
        ], dim=1)
        
        attention_mask = torch.cat([
            torch.ones(1, 1, device=DEVICE),
            tokenized.attention_mask
        ], dim=1)
        
        outputs = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=model.tokenizer.eos_token_id
        )
        
    return model.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure 3 channels (RGB)
        frames.append(frame)
    
    cap.release()
    
    # TimeSformer requires channels-first format
    transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # TimeSformer specific
    ])
    
    # Convert frames to tensor
    frames = [transform(Image.fromarray(f)) for f in frames]
    if len(frames) == 0:
        raise ValueError("No frames were loaded from the video.")
    
    # Stack frames into a tensor of shape [num_frames, 3, height, width]
    video_tensor = torch.stack(frames)  # [num_frames, 3, H, W]
    
    # Add batch dimension to make it [1, num_frames, 3, height, width]
    return video_tensor.unsqueeze(0).to(DEVICE)  # [1, num_frames, 3, H, W]

if __name__ == "__main__":
    train()
    
    # Load trained model
    model = SportVLM()
    model.load_state_dict(torch.load("trained_model/pytorch_model.bin"))
    model.to(DEVICE).eval()
    
    analysis = generate_analysis(model, "/Users/edoardobianchi/DATA_SCIENZE/CLIMBING/DynamicCropper/cam02.mp4")
    print("\n=== ANALYSIS ===")
    print(analysis)
