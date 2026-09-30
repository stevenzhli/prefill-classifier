import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


class GemmaSingleQuestionClassifier:
    def __init__(self, model_id="google/gemma-4-E2B-it", device="mps", dtype=torch.bfloat16):
        """
        Initializes the model and processor once, keeping them in memory for subsequent calls.
        """
        self.device = torch.device(device)
        print(f"Loading processor and model '{model_id}' to {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, payloads, classes, system_prompt, reasoning_tokens=0, temperature=1.0):
        """
        Processes a list of payloads sequentially (original unified-pass behavior).
        Returns Scikit-Learn compatible NumPy arrays of shape (n_samples, n_classes).
        
        :param payloads: A list of strings (for text) or dicts [{"text": "...", "image": PIL.Image}, ...]
        :param classes: A list of strings representing the exact classification categories.
        :param system_prompt: Base instructions for the model's persona/task.
        :param reasoning_tokens: Max reasoning tokens before locking choice.
        :param temperature: Float temperature to scale the logits.
        """
        # Ensure payloads is a list
        if not isinstance(payloads, list):
            payloads = [payloads]
            
        # 1. Format choices to ensure they have a leading space. 
        formatted_classes = [c if c.startswith(" ") else " " + c for c in classes]
        
        # Extract the single-token ID for each choice
        choice_token_ids = [
            self.processor.tokenizer(c, add_special_tokens=False)["input_ids"][0] 
            for c in formatted_classes
        ]

        # 2. Dynamically modify the system prompt based on reasoning mode
        trigger_string = "Label:"
        if reasoning_tokens > 0:
            system_prompt += (
                f"\nFirst, reason step-by-step. When your reasoning is complete, "
                f"you MUST output the exact string '{trigger_string}' followed immediately by your final choice."
            )
        else:
            system_prompt += "\nYou must output exactly one of the provided classes immediately. Do not provide any other text."
        
        # Storage for Scikit-Learn arrays
        all_logits = []
        all_probas = []
        all_preds = []
        all_pred_labels = []
        all_reasoning = []
        all_latency = []

        # 3. Iterate over the dataset
        for payload in tqdm(payloads):
            time1 = time.time()

            messages = [{"role": "system", "content": system_prompt}]
            # Parse the payload (Text vs Multimodal)
            images = None
            if isinstance(payload, dict) and "image" in payload:
                user_content = [
                    {"type": "image"},
                    {"type": "text", "text": payload.get("text", "")}
                ]
                images = [payload["image"]]
            else:
                text_payload = payload.get("text") if isinstance(payload, dict) else str(payload)
                user_content = text_payload

            # If reasoning is OFF, append trigger string to user prompt
            if reasoning_tokens == 0:
                if isinstance(user_content, list):
                    user_content[-1]["text"] += f"\n{trigger_string}"
                else:
                    user_content += f"\n{trigger_string}"

            messages.append({"role": "user", "content": user_content})

            # 4. Apply Chat Template and process inputs
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if images:
                inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

            reasoning_text = ""

            # 5. Handle Generation/Forward Pass
            if reasoning_tokens > 0:
                with torch.no_grad():
                    generated_sequence = self.model.generate(
                        **inputs,
                        max_new_tokens=reasoning_tokens,
                        tokenizer=self.processor.tokenizer,
                        stop_strings=[trigger_string],
                    )
                
                forward_inputs = {
                    "input_ids": generated_sequence,
                    "attention_mask": torch.ones_like(generated_sequence, device=self.device)
                }
                if "pixel_values" in inputs:
                    forward_inputs["pixel_values"] = inputs["pixel_values"]

                input_length = inputs["input_ids"].shape[-1]
                reasoning_text = self.processor.tokenizer.decode(generated_sequence[0][input_length:], skip_special_tokens=True)
            else:
                forward_inputs = inputs
        
            # 6. Single Forward Pass for Logits
            with torch.no_grad():
                outputs = self.model(**forward_inputs)

            time2 = time.time()

            # 7. Extract Logits for choices
            next_token_logits = outputs.logits[0, -1, :]
            choice_logits = next_token_logits[choice_token_ids]
            probabilities = F.softmax((choice_logits / temperature), dim=-1)

            best_idx = torch.argmax(choice_logits).item()
            best_label = classes[best_idx]
            
            all_logits.append(choice_logits.detach().cpu().tolist())
            all_probas.append(probabilities.detach().cpu().tolist())
            all_preds.append(best_idx)
            all_pred_labels.append(best_label)
            all_reasoning.append(reasoning_text.strip())
            all_latency.append(time2 - time1)

        # 8. Compile standard Scikit-Learn Numpy structures
        out = {
            "logits": np.array(all_logits, dtype=np.float32),        # shape: (n_samples, n_classes)
            "probas": np.array(all_probas, dtype=np.float32),        # shape: (n_samples, n_classes)
            "predictions": np.array(all_preds, dtype=np.int32),      # shape: (n_samples,)
            "prediction_labels": np.array(all_pred_labels),          # shape: (n_samples,)
            "latency": np.array(all_latency)
        }

        if reasoning_tokens > 0:
            out["reasoning_text"] = all_reasoning

        return out

