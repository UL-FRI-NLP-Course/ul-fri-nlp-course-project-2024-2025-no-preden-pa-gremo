import os
import time
import google.generativeai as genai
from bs4 import BeautifulSoup

default_custom_instructions = '''
LLM GaMS-9B-Instruct sem dal naslednja navodila:
[instructions]

In naslednje podatke o razmerah na cestah:
[data]

Vrnil mi je odgovor:
[gams_response]

Ali lahko oceniš kako dobro se je držal navodil? Lahko prešteješ, česa se je držal in česa se ni držal. Navodila, ki jih ni bilo potrebno upoštevati, ker taki podatki niso bili podani so lahko ignorirana. Glede na prešteto mi podaj tudi oceno od 0-1, kako dobrer je odgovor modela GaMS-9B-Instruct. Oceno podaj v formatu "Ocena: x.xx", zato da jo lahko avtomatsko preberemo iz odgovora. V odgovoru vrni le: navodila, ki se jih je držal in navodila ki se jih ni držal, ter na koncu oceno. odgovor naj bo kratek in jedernat.

Glede na oceno in navodila, ki se jih gams ni držal, popravi obstoječa navodila, tako da bo njegov naslednji odgovor boljši. Poskrbi, da se bo povsem držal navodil. Ta nova navodila mi pošlji v novi vrstici, čisto na koncu odgovora. Ne dodajaj drugega teksta, le nova navodila v zadnjem odstavku. Na začetku navodil nujno uporabi znak $, da jih lahko avtomatsko izločimo iz odgovora.
'''

# --- Configuration ---
GOOGLE_API_KEY = "SIMISLUJA"
MODEL_NAME = "gemini-2.5-flash-preview-04-17"

INSTRUCTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Instructions", "instructions.txt")

#with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
#    custom_instructions = f.read()

#print(custom_instructions)

MAX_GENERATION_TOKENS = 9000  # Adjust as needed

# --- Initialize Gemini API ---
try:
    genai.configure(api_key="[INSERT_YOUR_KEY_HERE]")
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Gemini API initialized successfully. Using model: {MODEL_NAME}")

except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    print(
        "Check your GOOGLE_API_KEY, internet connection, and library version."
    )
    print(
        "Make sure you have the latest google-generativeai: pip install --upgrade google-generativeai"
    )
    exit()

# --- Interactive Loop ---
chat = model.start_chat(history=[])

print("\nStarting interactive chat session with Gemini.")
print("Type 'quit', 'exit', or 'stop' to end the session.")
print("-" * 30)

'''while True:
    try:
        user_input = input("You: ")
        user_input_cleaned = user_input.lower().strip()

        if user_input_cleaned in ["quit", "exit", "stop"]:
            print("Exiting chat session.")
            break
        if not user_input.strip():
            continue

        # --- Clean HTML Tags from User Input ---
        soup = BeautifulSoup(user_input, "html.parser")
        text_only = soup.get_text()
        # --- end cleaning

        print("Model: Thinking...")
        start_gen_time = time.time()

        # --- Call the Gemini API ---
        try:
            response = chat.send_message(
                f"{custom_instructions}\n{text_only}",
                stream=False,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=MAX_GENERATION_TOKENS,
                    temperature=0.1,
                ),
            )
            end_gen_time = time.time()
            #print(response)
            # --- Extract the generated text ---
            gemini_output = response.text.strip()

            print(f"Model: {gemini_output}")
            print(
                f"(Generation took: {end_gen_time - start_gen_time:.2f} seconds)"
            )
            print("-" * 10)

        except Exception as e:
            print(f"\nError during Gemini API call: {e}")
            if "API key" in str(e):
                print("Check your GOOGLE_API_KEY.")
            print("Please try again or type 'quit' to exit.")

    except EOFError:
        print("\nExiting due to EOF.")
        break
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt.")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please try again or type 'quit' to exit.")
'''


def chat_with_gemini(instructions, data, gams_response):
    print("Gemini: Thinking...")
    start_gen_time = time.time()

    prompt = default_custom_instructions.replace("[instructions]", instructions).replace("[gams_response]", gams_response).replace("[data]", data)

    #print("PROMPT:\n" + prompt)

    # --- Call the Gemini API ---
    try:
        response = chat.send_message(
            f"{instructions}\n{prompt}",
            stream=False,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_GENERATION_TOKENS,
                temperature=0.1,
            ),
        )
        end_gen_time = time.time()
        # print(response)
        # --- Extract the generated text ---
        gemini_output = response.text.strip()

        #print(f"Model: {gemini_output}")
        print(
            f"(Generation took: {end_gen_time - start_gen_time:.2f} seconds)"
        )
        print(("-" * 50) + "\n")

    except Exception as e:
        print(f"\nError during Gemini API call: {e}")
        if "API key" in str(e):
            print("Check your GOOGLE_API_KEY.")
        print("Please try again or type 'quit' to exit.")
    except EOFError:
        print("\nExiting due to EOF.")
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please try again or type 'quit' to exit.")

    return gemini_output


print("\n--- Script Finished ---")
