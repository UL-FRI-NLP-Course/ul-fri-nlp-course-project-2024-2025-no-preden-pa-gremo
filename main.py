import threading
import itertools
import sys
import time
from datetime import datetime
from Data.readData import get_final_traffic_text  # import the function
from LLMs.gaMS import chat_with_gams
from LLMs.gemy import chat_with_gemini


default_custom_instructions = '''
Na podlagi vhodnih podatkov oblikuj kratke, informativne prometne novice, kot bi jih prebral poslušalec radia. Uporabi spodnja pravila:

1. NUJNO - Novice po nujnosti beri v naslednjem zaporedju:
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

1. Struktura povedi:
   - Cesta in smer + razlog + posledica in odsek
   - ali: Razlog + cesta in smer + posledica in odsek

2. Poimenuj avtoceste in smeri dosledno (primeri):
   - PRIMORSKA AVTOCESTA: proti Kopru / proti Ljubljani
   - DOLENJSKA AVTOCESTA: proti Obrežju / proti Ljubljani
   - VIPAVSKA HITRA CESTA: med razcepom Nanos in Ajdovščino
   - OBALNA HITRA CESTA: razcep Srmin – Izola
   - Opiši ljubljansko obvoznico po krakih: vzhodna, zahodna, severna, južna

3. Zastojev NE objavljaj, če:
   - So kratki (do 1 km) in posledica zgolj gostega prometa
   - So posledica običajnih prometnih konic

4. Posebna pravila:
   - Ob zaporah navedi obvoz: "Obvoz je po vzporedni regionalni cesti ..."
   - Ob koncu dogodka napiši OPOVED: "Promet znova poteka brez ovir ..."
   - Pri burji natančno označi stopnjo in omejitve prometa (glej primere spodaj)
   - Vedno navedi širši odsek (med dvema priključkoma), zlasti pri predorih in počivališčih

'''

def spinner(text, stop_event):
    for c in itertools.cycle('|/-\\'):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{text} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(text) + 2) + '\r')  # Clear line

def step_one():
    global default_custom_instructions

    while True:
        date_input = input("Enter date (YYYY-MM-DD HH:MM:SS): ")
        print() 
        if date_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            return None
        try:
            user_date = datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
            # Spinner setup
            stop_event = threading.Event()
            spinner_thread = threading.Thread(target=spinner, args=("Reading news from excel", stop_event))
            spinner_thread.start()
            # Call get_final_traffic_text with the input date string
            traffic_report = get_final_traffic_text(date_input)
            stop_event.set()
            spinner_thread.join()
            if traffic_report is None:
                print("There are no reports for the inputted time.")
                continue  # ask for date again
            else:
                print("Vhodni podatki:\n" + traffic_report)
                print(("-" * 50) + "\n")
                while True:
                    gams_response = chat_with_gams(traffic_report, default_custom_instructions)
                    print(f"GaMS: {gams_response}")
                    print(("-" * 50) + "\n")

                    response_gemini = chat_with_gemini(default_custom_instructions, traffic_report, gams_response)
                    new_instructions = response_gemini.split("$")[1]

                    print(f"Gemini: {response_gemini.split("$")[0]}")

                    if input("Do you want Gemini to generate new instructions? (y/n) ").lower()  == "y":
                        default_custom_instructions = new_instructions
                        print(f"New instructions: {response_gemini.split("$")[1]}")
                    else:
                        break

                if input("Do you want to generate a report for a new date? (y/n) ").lower() == "n":
                    break

                return user_date
        except ValueError:
            print("Invalid format. Please enter date as YYYY-MM-DD HH:MM:SS.")

def main():
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("------------- News Generation Tool -------------")
    print("Welcome to the news generation tool! \n"
          "Type 'exit' or 'quit' to stop. \n"
          "Type 'next' to generate a new news article.")
    print("------------------------------------------------")

    while True:
        user_date = step_one()
        if user_date is None:
            break

        while True:
            user_input = input("Command input (quit/next):")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                return
            if user_input.lower() == "next":
                break
            print(f"Echo: {user_input}")

if __name__ == "__main__":
    main()