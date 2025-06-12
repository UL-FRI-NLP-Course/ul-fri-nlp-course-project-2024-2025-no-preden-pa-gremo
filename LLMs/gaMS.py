import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import os
import time

# --- Configuration ---
model_id = "cjvt/GaMS-2B-Instruct"
#model_id = "google/gemma-7b-it"

use_quantization_if_gpu = True

# --- Device Detection (Prioritize CUDA) ---
# (GPU/CPU detection code remains the same)
pipeline_device = -1
compute_dtype = None
if torch.cuda.is_available():
    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    pipeline_device = 0
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

# --- Quantization Setup (GPU Only) ---
# (Quantization setup code remains the same - using model_kwargs)
quantization_config = None
model_kwargs = {}
if pipeline_device == 0 and use_quantization_if_gpu:
    try:
        import bitsandbytes
        import accelerate
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        compute_dtype = None
        print("Attempting 4-bit quantization (NF4) via bitsandbytes.")
        print("Requires 'bitsandbytes' and 'accelerate' libraries.")
    except ImportError:
        print("WARNING: 'bitsandbytes' or 'accelerate' library not found.")
        print("Install them (`pip install bitsandbytes accelerate`) for 4-bit quantization.")
        print("Proceeding without quantization.")
        quantization_config = None
        use_quantization_if_gpu = False
print("-" * 30)

# --- Load Tokenizer ---
# (Tokenizer loading code remains the same)
print(f"Loading tokenizer: {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer loaded. EOS token ID: {eos_token_id}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()
print("-" * 30)

# --- Initialize Pipeline ---
# (Pipeline initialization code remains the same - trust_remote_code direct argument)
print(f"Initializing text-generation pipeline for {model_id}...")
try:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=compute_dtype,
        trust_remote_code=True, # Direct argument
        model_kwargs=model_kwargs
    )
    print("Pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    # ... (error handling) ...
    exit()
print("-" * 30)

# --- Interactive Loop ---
message_history = [] # Stores the conversation history

'''
print("\nStarting interactive chat session.")
print("Type 'quit', 'exit', or 'stop' to end the session.")
print("-" * 30)

while True:
    try:
        user_input = input("You: ")
        user_input_cleaned = user_input.lower().strip()

        if user_input_cleaned in ["quit", "exit", "stop"]:
            print("Exiting chat session.")
            break
        if not user_input.strip():
            continue

        # --- MODIFICATION: Prepend instructions to first user message content ---
        current_message_content = user_input

        # Check if this is the very first turn AND custom instructions exist
        if not message_history and custom_instructions:
             # Combine instructions and user input for the first message
             current_message_content = f"{custom_instructions}\n\n{user_input}"
             print("(Prepending custom instructions to this first message)") # Optional log

        # Add the (potentially modified) user message to the history
        message_history.append({"role": "user", "content": current_message_content})
        # --- END MODIFICATION ---

        print(f"System: Thinking... (on {'GPU' if pipeline_device == 0 else 'CPU'})")
        start_gen_time = time.time()

        # --- Call the pipeline with the current message history ---
        # The pipeline will apply the tokenizer's chat template to the history
        response = pipe(
            message_history, # Pass the history directly
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token_id,
            # pad_token_id=eos_token_id # Add if needed for batching/padding issues
        )

        end_gen_time = time.time()

        # --- Extract response and update history ---
        # (Response extraction logic remains the same)
        try:
            full_generated_output_list = response[0]["generated_text"]
            new_assistant_message = full_generated_output_list[-1]

            if new_assistant_message["role"] == "assistant":
                assistant_response_content = new_assistant_message["content"]
                # IMPORTANT: Add the *actual* assistant response to history
                message_history.append(new_assistant_message)

                print(f"Model: {assistant_response_content}")
                print(
                    f"(Generation took: {end_gen_time - start_gen_time:.2f} seconds)"
                )
            else:
                print("\nError: Expected assistant message at the end of pipeline output, but got:")
                print(new_assistant_message)
                # Roll back the user message we added for this failed turn
                if message_history and message_history[-1]["role"] == "user":
                    message_history.pop()
                    print("Removed last user message from history due to unexpected response format.")

        except (IndexError, KeyError, TypeError) as e:
            print(f"\nError parsing pipeline response: {e}")
            print("Raw response object:", response)
            # Roll back the user message we added for this failed turn
            if message_history and message_history[-1]["role"] == "user":
                message_history.pop()
                print("Removed last user message from history due to parsing error.")

        print("-" * 10)

    # (Exception handling code remains the same)
    except EOFError:
        print("\nExiting due to EOF.")
        break
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt.")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # ... (error handling) ...
        print("Please try again or type 'quit' to exit.")

'''
def chat_with_gams(prompt, instructions):
    """
    Chat with GaMS model, maintaining conversation history across calls.
    Args:
        instructions (str): Custom instructions to prepend on the first turn.
        prompt (str): User's input message.
    Returns:
        str: Assistant's response.
    """

    print("Running gams...")

    #if not hasattr(chat_with_gams, "message_history"):
    chat_with_gams.message_history = []

    message_history = chat_with_gams.message_history

    if not message_history and instructions:
        current_message_content = f"{instructions}\n\n{prompt}"
    else:
        current_message_content = prompt

    message_history.append({"role": "user", "content": current_message_content})

    try:
        response = pipe(
            message_history,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            eos_token_id=eos_token_id,
        )

        full_generated_output_list = response[0]["generated_text"]
        new_assistant_message = full_generated_output_list[-1]

        if new_assistant_message["role"] == "assistant":
            assistant_response_content = new_assistant_message["content"]
            message_history.append(new_assistant_message)
            return assistant_response_content
        else:
            if message_history and message_history[-1]["role"] == "user":
                message_history.pop()
            raise ValueError("Unexpected response format from model.")

    except Exception as e:
        if message_history and message_history[-1]["role"] == "user":
            message_history.pop()
        raise e

print("\n--- Script Finished ---")


