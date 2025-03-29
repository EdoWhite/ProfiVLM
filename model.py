import torch
import av
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TimesformerModel, TrainingArguments, Trainer, AutoImageProcessor
)
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from peft import LoraConfig, get_peft_model
import json
import os
from functools import partial

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8
LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M-Instruct"
VISION_ENCODER = "facebook/timesformer-base-finetuned-k600"

class SportVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT)
        self._add_special_tokens()
        
        # Load and configure LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_CHECKPOINT,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2",
        )
        self._configure_adapters()
        
        # Vision encoder
        self.vision_encoder = TimesformerModel.from_pretrained(
            VISION_ENCODER,
            num_frames=NUM_FRAMES
        ).requires_grad_(False)
        self.vision_processor = AutoImageProcessor.from_pretrained(VISION_ENCODER)
        
        # Projector
        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(768, self.llm.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.llm.config.hidden_size),
            torch.nn.Dropout(0.1)
        )
        
        self.print_trainable_parameters()
    
    def _add_special_tokens(self):
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<video>"]})
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<video>")
        
    def _configure_adapters(self):
        # Freeze base model
        self.llm.requires_grad_(False)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, video, input_ids, attention_mask, labels):
        print("Forward pass")
        # Video processing
        inputs = self.vision_processor(video, return_tensors="pt")
        video_features = self.vision_encoder(**inputs).last_hidden_state
        video_emb = self.video_adapter(video_features)
        
        # Get text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)
        
        # Replace video tokens with video embeddings
        video_mask = (input_ids == self.video_token_id)
        text_emb[video_mask] = video_emb
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def print_trainable_parameters(self):
        trainable = 0
        total = 0
        for _, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"Trainable: {trainable:,} | Total: {total:,} | %: {100*trainable/total:.2f}")

class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path):
        self.transform = Compose([
            Resize((224, 224)),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        
        with open(annotation_path) as f:
            self.annotations = []
            for line in f:
                try:
                    ann = json.loads(line)
                    assert all(k in ann for k in ["video_path", "analysis", "proficiency_level"])
                    self.annotations.append(ann)
                except Exception as e:
                    print(f"Skipping invalid entry: {str(e)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        try:
            container = av.open(ann["video_path"])
            indices = sample_frame_indices(clip_len=NUM_FRAMES, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)
            return {
                'video': video,
                'text': f"<video>\nAnalysis: {ann['analysis']}\nProficiency: {ann['proficiency_level']}"
            }
        except Exception as e:
            print(f"Error processing {ann['video_path']}: {str(e)}")
            return {
                'video': torch.zeros((1, 3, NUM_FRAMES, 224, 224)),
                'text': "<video>\nAnalysis: Error\nProficiency: Error"
            }

def collate_fn(examples, tokenizer):
    videos = []
    texts = []
    
    for ex in examples:
        videos.append(ex['video'])
        texts.append([{
            "role": "user", 
            "content": "Analyze the sports performance video: <video>"
        }, {
            "role": "assistant",
            "content": ex['text'].split("\n", 1)[1]
        }])
    
    # Process videos
    #video_tensors = torch.cat([v for v in videos])
    #print(f"Batch video shape: {video_tensors.shape}")
    
    # Process texts
    batch = tokenizer.apply_chat_template(
        texts,
        add_generation_prompt=False,
        padding=True,
        return_tensors="pt",
        return_dict=True
    )
    
    # Mask video tokens in labels
    labels = batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[labels == tokenizer.convert_tokens_to_ids("<video>")] = -100
    
    return {
        "video": videos,
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels
    }

def train():
    model = SportVLM()
    
    # Dataset setup
    train_dataset = VideoTextDataset("/data/users/edbianchi/ProfiVLM/annotations/annotation.jsonl")
    if len(train_dataset) == 0:
        raise ValueError("No valid training samples found!")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(collate_fn, tokenizer=model.tokenizer),
    )
    
    trainer.train()
    model.save_pretrained("trained_models/sportvlm_complete")

def generate_analysis(model, video_path):
    model.eval()
    try:
        video = load_video(video_path).to(DEVICE)
    except Exception as e:
        return f"Error loading video: {str(e)}"
    
    with torch.no_grad():
        # Process video
        video_features = model.vision_encoder(video).last_hidden_state.mean(dim=1)
        video_emb = model.video_adapter(video_features)
        
        # Prepare prompt
        messages = [{
            "role": "user",
            "content": "Analyze the sports performance video: <video>"
        }]
        
        inputs = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(DEVICE)
        
        # Replace video token
        video_mask = (inputs["input_ids"] == model.video_token_id)
        inputs_embeds = model.llm.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds[video_mask] = video_emb
        
        # Generate
        outputs = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=model.tokenizer.eos_token_id
        )
    
    return model.tokenizer.decode(outputs[0], skip_special_tokens=True)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, NUM_FRAMES, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {idx}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    # Process frames
    transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    # Return shape: [1, C, T, H, W]
    ret = torch.stack([
        transform(Image.fromarray(f)) for f in frames
    ]).unsqueeze(0).permute(0, 1, 2, 3, 4)
    
    print(f"Loaded video with shape {ret.shape}")
    return ret

if __name__ == "__main__":
    train()
    
    # Load model
    model = SportVLM.from_pretrained("trained_models/sportvlm_complete")
    model.to(DEVICE).eval()
    
    analysis = generate_analysis(model, "/data/users/edbianchi/ProEstVideo/cam02.mp4")
    print("\n=== ANALYSIS ===")
    print(analysis.split("[/INST]")[-1].strip())