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
from safetensors.torch import save_model, load_model
from functools import partial


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8
LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M-Instruct"

class SportVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_CHECKPOINT, 
                    torch_dtype=torch.bfloat16, 
                    cache_dir = "/data/users/edbianchi/.cache",
                    device_map="auto",
                    _attn_implementation="flash_attention_2",
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
        
        # Apply LoRA
        self.llm = get_peft_model(self.llm, lora_config)

        # Freeze vision encoder
        self.vision_encoder = TimesformerModel.from_pretrained(
                    "facebook/timesformer-base-finetuned-k600",
                    num_frames = NUM_FRAMES,
                    cache_dir = "/data/users/edbianchi/.cache"
            )
        self.vision_encoder.requires_grad_(False)

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<video>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Video projector
        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, self.llm.config.hidden_size),
            torch.nn.Dropout(0.1)
        )
        
        self.print_trainable_parameters()
        
    def print_trainable_parameters(self):
        trainable = 0
        total = 0
        for _, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"Trainable: {trainable:,} | Total: {total:,} | %: {100*trainable/total:.2f}")


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
    def __init__(self, annotation_path):
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
            video = torch.zeros((1, NUM_FRAMES, 3, 224, 224))
            if os.path.exists(ann["video_path"]):
                try:
                    video = load_video(ann["video_path"])
                except Exception as e:
                    print(f"Error loading {ann['video_path']}: {str(e)}")
            
            # Ensure video tensor has correct shape
            if video.dim() != 5 or video.size(2) != 3:
                raise ValueError(f"Invalid video tensor shape: {video.shape}")
            
            return {
                'video': video.squeeze(0),  # [T, C, H, W]
                'text': f"Analysis: {ann['analysis']}\nProficiency: {ann['proficiency_level']}"
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return {
                'video': torch.zeros((NUM_FRAMES, 3, 224, 224)),
                'text': "Analysis: Unknown\nProficiency: Unknown"
            }

def collate_fn(examples, tokenizer):
    messages_list = []
    videos = []
    for example in examples:
        question = "Classify the athlete's proficiency level and briefly justify your answer."
        analysis = example['text']
        messages = [
            {"role": "user", "content": f"{question}\n<video>"},
            {"role": "assistant", "content": analysis}
        ]
        messages_list.append(messages)
        videos.append(example['video'])

    encoded = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        padding=True
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    video_token_id = tokenizer.convert_tokens_to_ids("<video>")
    labels[labels == video_token_id] = -100

    return {
        "video": torch.stack(videos),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train():
    model = SportVLM().to(DEVICE)
    
    # Initialize dataset with validation
    train_dataset = VideoTextDataset(
        annotation_path="/data/users/edbianchi/ProfiVLM/annotations/annotation.jsonl"
    )
    
    # Verify dataset
    if len(train_dataset) == 0:
        raise ValueError("No valid training samples found!")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=10,  # Reduce batch size to lower memory usage
        learning_rate=1e-3,
        warmup_ratio=0.1,
        gradient_accumulation_steps=3,  # Reduce gradient accumulation steps
        fp16=torch.cuda.is_available(),  # Enable mixed precision
        logging_steps=2,
        save_strategy="epoch",
        remove_unused_columns=False,
        save_safetensors=False,
        )
    
    data_collator = partial(collate_fn, tokenizer=model.tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    torch.save(model.state_dict(), "trained_models/sportvlm_complete.pth")
    print("Model saved to trained_models/sportvlm_complete.pth")
    
def generate_analysis(model, video_path):
    try:
        video = load_video(video_path).to(DEVICE)
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        return "Video analysis failed: invalid input"

    with torch.no_grad():
        video_features = model.vision_encoder(video).last_hidden_state.mean(dim=1)
        video_emb = model.video_adapter(video_features)

        messages = [
            {
                "role": "user",
                "content": "Analyze the following sports video. Provide a proficiency label and justify your answer.\n<video>"
            }
        ]

        prompt = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(DEVICE)

        text_emb = model.llm.get_input_embeddings()(prompt["input_ids"])

        inputs_embeds = torch.cat([
            video_emb.unsqueeze(1).to(model.llm.dtype),
            text_emb
        ], dim=1)

        attention_mask = torch.cat([
            torch.ones(1, 1, device=DEVICE),
            prompt["attention_mask"]
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

    return model.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    return video_tensor.unsqueeze(0)  # [1, num_frames, 3, H, W]

if __name__ == "__main__":
    train()
    
    # Load trained model
    print("Initializing model...")
    model = SportVLM()
    print("Loading model weights...")
    model.load_state_dict(torch.load("trained_models/sportvlm_complete.pth"))
    model.to(DEVICE).eval()
    print("Model loaded successfully.")
    print("Generating analysis...")
    
    analysis = generate_analysis(model, "/data/users/edbianchi/ProEstVideo/cam02.mp4")
    print("\n=== ANALYSIS ===")
    print(analysis)
