import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import os
import time
from docx import Document


# --- Configuration ---
model_id = "cjvt/GaMS-9B-Instruct"
promet_docx_path = "./RTVSlo/PROMET.docx"

custom_instructions = ""
use_quantization_if_gpu = True

try:
   # Read the .docx file
    doc = Document(promet_docx_path)
    # Combine all paragraphs into a single string
    custom_instructions = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    print(f"Custom instructions loaded from {promet_docx_path}.")
except Exception as e:
    print(f"Error reading {promet_docx_path}: {e}")
    print("Using default custom instructions.")

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

# --- Print Custom Instructions ---
if custom_instructions:
    print("-" * 30)
    print("Using Custom Instructions (Prepended to first user message):")
    print(custom_instructions)
else:
    print("-" * 30)
    print("No custom instructions provided.")
print("-" * 30)

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

        custom_instructions = '''
        Navodila za generiranje prometnih informacij v slovenščini za radijsko postajo:

        "Izhodni format:

        Podatki o prometu:

        [Informacija 1]

        [Informacija 2]

        [Informacija 3]

        ...

        Poimenovanje avtocest in smeri. Poskrbi, da boš uporabil pravilne smeri in imena avtocest. Uporabi naslednje smeri in imena avtocest:

        * Ljubljana-Koper: Primorska avtocesta / proti Kopru / proti Ljubljani
        * Ljubljana-Obrežje: Dolenjska avtocesta / proti Obrežju / proti Ljubljani
        * Ljubljana-Karavanke: Gorenjska avtocesta / proti Karavankam ali Avstriji / proti Ljubljani
        * Ljubljana-Maribor: Štajerska avtocesta / proti Mariboru / proti Ljubljani
        * Maribor-Lendava: Pomurska avtocesta / proti Mariboru / proti Lendavi ali Madžarski
        * Maribor-Gruškovje: Podravska avtocesta / proti Mariboru / proti Gruškovju ali Hrvaški (nikoli proti Ptuju)
        * Razcep Gabrk – Fernetiči: proti Italiji / proti primorski avtocesti, Kopru, Ljubljani (ni primorska avtocesta)
        * Maribor-Šentilj (mejni prehod Šentilj - razcep Dragučova): od Maribora proti Šentilju / od Šentilja proti Mariboru (ni štajerska avtocesta)
        * Mariborska vzhodna obvoznica (razcep Slivnica - razcep Dragučova): proti Avstriji ali Lendavi / proti Ljubljani (nikoli proti Mariboru)
        * Hitra cesta skozi Maribor: Regionalna cesta Betnava-Pesnica / NEKDANJA hitra cesta skozi Maribor (ne "BIVŠA hitra cesta skozi Maribor")
        * Ljubljanska obvoznica:
            * Vzhodna (razcep Malence proti Novemu mestu - razcep Zadobrova proti Mariboru)
            * Zahodna (razcep Koseze proti Kranju - razcep Kozarje proti Kopru)
            * Severna (razcep Koseze proti Kranju - razcep Zadobrova proti Mariboru)
            * Južna (razcep Kozarje proti Kopru - razcep Malence proti Novemu mestu)
        * Razcep Nanos-Vrtojba: Vipavska hitra cesta / proti Italiji ali Vrtojbi / proti Nanosu ali primorski avtocesti ali Razdrtemu (nikoli "primorska hitra cesta")
        * Razcep Srmin-Izola: Obalna hitra cesta / proti Kopru ali Portorožu (nikoli "primorska hitra cesta")
        * Koper-Škofije: Na hitri cesti od Kopra proti Škofijam / na hitri cesti od Škofij proti Kopru (smer je že vključena)
        * Mejni prehod Dolga vas-Dolga vas: Na hitri cesti od mejnega prehoda Dolga vas proti pomurski avtocesti / na hitri cesti proti mejnemu prehodu Dolga vas (zelo redko)
        * ŠKOFJA LOKA – GORENJA VAS: Regionalna cesta proti Ljubljani / proti Gorenji vasi (pogovorno škofjeloška obvoznica, pomembno zaradi zaprtega predora Stén)
        * Ljubljana-Črnuče – Trzin: Glavna cesta od Ljubljane proti Trzinu / od Trzina proti Ljubljani (včasih "trzinska obvoznica")
        * Končne destinacije namesto vmesnih krajev: Proti Avstriji/Karavankam, proti Hrvaški/Obrežju/Gruškovju, proti Madžarski... (namesto proti Kranju, Novemu mestu, Ptuju, Murski Soboti)

        Struktura prometne informacije. Poskrbi za pravilno strukturo in formatiranje prometnih informacij. Uporabi naslednje smernice:

        * Cesta in smer + razlog + posledica in odsek
        * Razlog + cesta in smer + posledica in odsek
        * A = avtocesta, H = hitra cesta, G = glavna cesta, R = regionalna cesta, L = lokalna cesta

        Nujne prometne informacije (objaviti vsakih 15-20 minut, posodabljati):

        * Zaprta avtocesta
        * Nesreča z zastojem na avtocesti, glavni ali regionalni cesti
        * Daljši zastoji (ne glede na vzrok, vsaj 1 km izven prometnih konic)
        * Pokvarjeno vozilo, ki zapira prometni pas
        * Voznik v napačni smeri
        * Pešci/živali/predmeti na vozišču (živali in predmete se lahko izloči po dogovoru)

        Zastoji:

        * Preveriti vzrok (dela, nesreča).
        * Ne objavljati zastojev krajših od 1 km (razen če se pričakuje daljšanje ali je povezano z dogodkom).
        * V prometnih konicah objaviti le nenavadno dolge zastoje (zjutraj Štajerska, popoldne severna/južna ljubljanska obvoznica).

        Hierarhija dogodkov. Tukaj je zelo pomembno, da se upošteva hierarhija dogodkov. Na splošno velja, da je treba najprej objaviti zaporo avtoceste, nato pa še vse ostale dogodke. Vendar pa je treba upoštevati tudi druge dejavnike, kot so dolžina zastoja in resnost dogodka:

        1. Voznik v napačno smer
        2. Zaprta avtocesta
        3. Nesreča z zastojem na avtocesti
        4. Zastoji zaradi del na avtocesti (nevarnost naletov)
        5. Zaradi nesreče zaprta glavna ali regionalna cesta
        6. Nesreče na avtocestah in drugih cestah
        7. Pokvarjena vozila (zaprt pas)
        8. Žival na vozišču
        9. Predmet/razsut tovor na avtocesti
        10. Dela na avtocesti (nevarnost naleta)
        11. Zastoj pred Karavankami in mejnimi prehodi

        Opozorila lektorjev:

        * Počasni pas = pas za počasna vozila.
        * Polovična zapora = izmenično enosmerno.
        * "Zaprta je polovica avtoceste" -> "Promet poteka le po polovici avtoceste v obe smeri."
        * Pokriti vkopi = predori (razen galerije Moste).
        * Razcepi: Navesti smer prihoda in odhoda (npr., "Na razcepu Kozarje je zaradi nesreče oviran promet iz smeri Viča proti Brezovici").
        * Predori/počivališča: Navesti širši odsek (med dvema priključkoma).
        * Obvozi: "Obvoz je po vzporedni regionalni cesti/po cesti Lukovica-Blagovica" ali "Vozniki se lahko preusmerijo na vzporedno regionalno cesto" (alternativni obvozi: "Vozniki SE LAHKO PREUSMERIJO TUDI, …").

        Formulacije:

        * Voznik v napačni smeri: "Opozarjamo voznike na [cesta in smer], da je na njihovo polovico zašel voznik v napačni smeri. Vozite skrajno desno in ne prehitevajte. ODPOVED je nujna!" / "Promet na [cesta in smer] ni več ogrožen zaradi voznika v napačni smeri."
        * Nesreča: "Prosimo voznike, naj se razvrstijo na skrajni levi in desni rob vozišča/odstavni pas za intervencijska vozila!" (ODPOVED po koncu zastojev!)

        Burja:

        * Stopnja 1: "Zaradi burje je na [cesta in odsek] prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton."
        * Stopnja 2: "Zaradi burje je na [cesta in odsek] prepovedan promet za hladilnike in vsa vozila s ponjavami."
        * Preklic: "Na [cesta in odsek] ni več prepovedi prometa zaradi burje." / "Na [cesta] je promet znova dovoljen za vsa vozila."

        Prepoved prometa:

        * "Do 21. ure velja prepoved prometa tovornih vozil nad 7,5 ton." / "Od 8. do 21. ure velja prepoved prometa tovornih vozil nad 7,5 ton, na primorskih cestah do 22. ure."

        Ne omenjaj da slediš navodilom. Kakršnekoli angleške podatke prevedi v slovenski jezik. Pretvarjaj se da si radiskij obveščevalec prometa v sloveniji. Nujno čisto natančno upoštevaj zgornja navodila in iz naslednjih podatkov sestavi novico ki bo prebrana na radiju, lepo formatiraj tekst:
        '''

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


print("\n--- Script Finished ---")
