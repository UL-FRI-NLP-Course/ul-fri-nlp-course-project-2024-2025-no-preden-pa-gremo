import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig

model_id = "google/gemma-7b-it"
use_quantization_if_gpu = True
pipeline_device = -1  # -1 = CPU for HF pipeline
compute_dtype = None

if torch.cuda.is_available():
    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    pipeline_device = 0  # use CUDA:0
    # Gemma supports bf16 & fp16
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Using device: cuda (GPU 0) with bfloat16")
    else:
        compute_dtype = torch.float16
        print("Using device: cuda (GPU 0) with float16")
else:
    print("NVIDIA GPU not found or CUDA not set up correctly.")
    print("Using device: cpu. This will be significantly slower.")
    use_quantization_if_gpu = False

print(f"Using model: {model_id}")

# --------------------------------------------------
# Optional 4‑bit quantization (GPU only, bitsandbytes)
# --------------------------------------------------
quantization_config = None
model_kwargs = {}
if pipeline_device == 0 and use_quantization_if_gpu:
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        # Let bitsandbytes manage dtype internally when quantising
        compute_dtype = None
        print("Attempting 4‑bit quantization (NF4) via bitsandbytes.")
    except ImportError:
        print("bitsandbytes or accelerate not installed – running without quantization.")
        use_quantization_if_gpu = False
print("-" * 30)

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------
print(f"Loading tokenizer: {model_id}…")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True  # Gemma ships its own chat template
    )
    eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer loaded. EOS token ID: {eos_token_id}")
except Exception as e:
    raise RuntimeError(f"Error loading tokenizer: {e}")
print("-" * 30)

# --------------------------------------------------
# Initialise text‑generation pipeline
# --------------------------------------------------
print(f"Initializing text‑generation pipeline for {model_id}…")
try:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    print("Pipeline initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Error initializing pipeline: {e}")
print("-" * 30)

# --------------------------------------------------
# Conversation helpers
# --------------------------------------------------

def _history():
    """Attach a persistent chat history list to this function object."""
    if not hasattr(_history, "store"):
        _history.store = []
    return _history.store


def chat_with_gemma(prompt: str, instructions: str = ""):
    print("Running Gemma…")
    history = _history()

    # Prepend custom instructions only on very first turn
    if not history and instructions:
        prompt = f"{instructions}\n\n{prompt}"

    history.append({"role": "user", "content": prompt})

    try:
        response = pipe(
            history,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token_id,
        )

        assistant_msg = response[0]["generated_text"][-1]
        if assistant_msg["role"] != "assistant":
            # Roll back if Gemma replies with unexpected schema
            history.pop()
            raise ValueError("Unexpected response format from model.")

        history.append(assistant_msg)
        return assistant_msg["content"]

    except Exception as err:
        # Clean state so future calls still work
        if history and history[-1]["role"] == "user":
            history.pop()
        raise RuntimeError(f"Inference failed: {err}") from err


def chat_with_gemma_stateless(prompt: str, instructions: str = ""):
    print("Running Gemma (stateless)...")

    temp_history = []

    if instructions:
        full_prompt = f"{instructions}\n\n{prompt}"
    else:
        full_prompt = prompt

    temp_history.append({"role": "user", "content": full_prompt})

    try:
        response = pipe(
            temp_history,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token_id,
        )

        assistant_msg = response[0]["generated_text"][-1]
        if assistant_msg["role"] != "assistant":
            raise ValueError("Unexpected response format from model.")

        return assistant_msg["content"]

    except Exception as err:
        raise RuntimeError(f"Inference failed: {err}") from err


# --------------------------------------------------
# Optional interactive CLI (uncomment to use)
# --------------------------------------------------
"""
print("\nStarting interactive chat session.")
print("Type 'quit', 'exit', or 'stop' to end the session.")
print("-" * 30)
custom_instructions = input("(Optional) System instructions to prepend on first turn – press Enter to skip:\n> ")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower().strip() in {"quit", "exit", "stop"}:
            print("Exiting chat session.")
            break
        if not user_input.strip():
            continue

        reply = chat_with_gemma(user_input, custom_instructions)
        print(f"Gemma: {reply}\n" + "-" * 10)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting chat session.")
        break
"""

print("\n--- Script ready: use chat_with_gemma(prompt, instructions) ---")
