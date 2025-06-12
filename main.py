import threading
import itertools
import sys
import time
from datetime import datetime
from Data.readData import get_final_traffic_text, get_real_traffic_report, analyze_reports
from LLMs.gaMS import chat_with_gams
from LLMs.gemy import chat_with_gemini
from LLMs.gemma import chat_with_gemma
from Scores.bert import calculate_bert
from Scores.bleu import calculate_bleu
import pandas as pd
import re
import os
import csv

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


def parse_report(full_text: str, report_n: int):
    """
    Parses a block of text to extract the timestamp, generated report, and real report
    for a specific rank number.
    """
    pattern = re.compile(
        fr"---------- RANK {report_n} ----------\n"
        r"TIMESTAMP:\s*(.*?)\n"
        r"SIMILARITY SCORE:.*?\n"
        fr"--- GENERATED REPORT for RANK {report_n} \(from Excel\) ---\n"
        r"(.*?)\n"
        fr"--- REAL REPORT for RANK {report_n} \(from RTF\) ---\n"
        r"(.*)",
        re.DOTALL
    )
    match = pattern.search(full_text)

    if match:
        timestamp = match.group(1).strip()
        generated = match.group(2).strip()
        real = match.group(3).strip()

        if '----------' in real:
            real = real.split('----------')[0].strip()

        return generated, real, timestamp
    else:
        return None, None, None


def parse_gemini_ratings(text: str) -> dict:
    """
    Parses the Gemini response text to extract all rating categories.
    """
    ratings = {}
    categories = [
        "Slovnica", "Hierarhija dogodkov", "Sestava prometne informacije",
        "Poimenovanje avtocest", "Generalna"
    ]
    for category in categories:
        # This pattern looks for the category name, "ocena:", a number, a hyphen, and the rest of the line.
        pattern = re.compile(fr"{category} ocena:\s*(\d+)\s*-\s*(.*)")
        match = pattern.search(text)
        if match:
            score = match.group(1).strip()
            justification = match.group(2).strip()
            ratings[category] = f"{score} - {justification}"
        else:
            ratings[category] = "Not Found"
    return ratings


def run_automated_improvement(model_name: str, report_context: str):
    """
    Runs the automated instruction improvement and scoring process,
    logging detailed results to a text file in real-time.
    """
    global default_custom_instructions
    results_filename = f"detailed_log_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Outer loop for the top 10 reports
    for report_n in range(1, 11):
        print(f"\n{'=' * 20} PROCESSING REPORT RANK {report_n}/10 {'=' * 20}")

        traffic_report, optimal_traffic_report, timestamp = parse_report(report_context, report_n)

        if not traffic_report or not optimal_traffic_report:
            print(f"Could not parse data for RANK {report_n}. Skipping.")
            continue

        with open(results_filename, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"STARTING ANALYSIS FOR REPORT RANK: {report_n} | TIMESTAMP: {timestamp}\n")
            f.write(f"INPUT DATA:\n{traffic_report}\n")
            f.write(f"{'=' * 80}\n")

        current_instructions = default_custom_instructions

        # Inner loop for 5 iterations of instruction improvement
        for iteration in range(1, 6):
            print(f"--- Iteration {iteration}/5 ---")

            stop_event = threading.Event()
            spinner_text = f"Generating response with {model_name.upper()}"
            spinner_thread = threading.Thread(target=spinner, args=(spinner_text, stop_event))
            spinner_thread.start()

            if model_name == "gams":
                model_response = chat_with_gams(traffic_report, current_instructions)
            else:
                model_response = chat_with_gemma(traffic_report, current_instructions)

            stop_event.set()
            spinner_thread.join()

            bleu_score = calculate_bleu(model_response, optimal_traffic_report)
            _, _, bert_f1 = calculate_bert(model_response, optimal_traffic_report)
            print(f"  Scores -> BLEU: {bleu_score:.4f}, BERT F1: {bert_f1:.4f}")

            new_instructions = "Error: Could not generate new instructions."
            gemini_ratings = {}
            if iteration < 5:
                stop_event = threading.Event()
                spinner_text = "Analyzing and improving instructions with Gemini"
                spinner_thread = threading.Thread(target=spinner, args=(spinner_text, stop_event))
                spinner_thread.start()

                try:
                    response_gemini = chat_with_gemini(current_instructions, traffic_report, model_response)
                    new_instructions = response_gemini.split("$")[1].strip()
                    gemini_ratings = parse_gemini_ratings(response_gemini)
                    print("  Instructions improved for next iteration.")
                except IndexError:
                    new_instructions = "# WARNING: Could not parse new instructions from Gemini's response. Re-using previous set. #"
                    print(f"  {new_instructions}")
                except Exception as e:
                    new_instructions = f"# ERROR: An exception occurred during instruction improvement: {e} #"
                    print(f"  {new_instructions}")

                stop_event.set()
                spinner_thread.join()
            else:
                new_instructions = "# FINAL ITERATION: No new instructions generated. #"
                # Still get the final ratings
                response_gemini = chat_with_gemini(current_instructions, traffic_report, model_response)
                gemini_ratings = parse_gemini_ratings(response_gemini)

            log_block = f"""
--------------------------------------------------------------------------------
ITERATION: {iteration}
--------------------------------------------------------------------------------

--- MODEL OUTPUT ---
{model_response}

--- SCORES ---
BLEU Score: {bleu_score:.4f}
BERT F1-Score: {bert_f1:.4f}

--- GEMINI RATINGS ---
Slovnica:                     {gemini_ratings.get("Slovnica", "Not Found")}
Hierarhija dogodkov:          {gemini_ratings.get("Hierarhija dogodkov", "Not Found")}
Sestava prometne informacije: {gemini_ratings.get("Sestava prometne informacije", "Not Found")}
Poimenovanje avtocest:        {gemini_ratings.get("Poimenovanje avtocest", "Not Found")}
Generalna:                    {gemini_ratings.get("Generalna", "Not Found")}

--- IMPROVED INSTRUCTIONS FOR NEXT ITERATION ---
{new_instructions}
"""
            with open(results_filename, 'a', encoding='utf-8') as f:
                f.write(log_block)

            if "WARNING" not in new_instructions and "ERROR" not in new_instructions:
                current_instructions = new_instructions

    print(f"\n{'=' * 20} AUTOMATED PROCESSING COMPLETE {'=' * 20}")
    print(f"All scores and details logged to '{results_filename}'")


def step_one():
    filename = "analysis_report-[2023-01-01][2023-12-31].txt"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            report_context = f.read()
        print(f"Successfully read content from '{filename}'")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please run `find_good_reports()` first.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    while True:
        model_input = input("Enter model to test ('gams' or 'gemma', or 'exit' to quit): ")
        print()
        if model_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            return
        if model_input.lower() in ["gams", "gemma"]:
            run_automated_improvement(model_input.lower(), report_context)
            break
        else:
            print("Invalid model input. Please enter 'gams' or 'gemma'.")


def main():
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("--------- Instruction Improvement Tool ---------")
    print("This tool automates the process of iteratively")
    print("improving instructions for a language model.")
    print("------------------------------------------------")
    step_one()


def find_good_reports():
    start_analysis_date = "2023-01-01 00:00:00"
    end_analysis_date = "2023-12-31 23:59:59"
    report_quality_df = analyze_reports(start_analysis_date, end_analysis_date)
    if not report_quality_df.empty:
        report_quality_df = report_quality_df.sort_values(by='hash_similarity', ascending=False)
        report_quality_df.reset_index(drop=True, inplace=True)
        print("\n--- Ultra-Fast Analysis Results ---")
        print("\n--- Top 10 Best Matches ---")
        print(report_quality_df[['timestamp', 'hash_similarity']].head(10))
        start_date_str_for_filename = datetime.strptime(start_analysis_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        end_date_str_for_filename = datetime.strptime(end_analysis_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        output_filename = f"analysis_report-[{start_date_str_for_filename}][{end_date_str_for_filename}].txt"
        print(f"\nGenerating detailed text report to '{output_filename}'...")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(f"ANALYSIS REPORT\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Analyzed From: {start_analysis_date} to {end_analysis_date}\n")
                f.write(f"Number of reports analyzed: {len(report_quality_df)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("--- TOP 100 BEST MATCHES (HIGHEST SIMILARITY) ---\n")
                f.write("=" * 80 + "\n")
                top_100_best = report_quality_df.head(100)
                for index, row in top_100_best.iterrows():
                    f.write(f"\n---------- RANK {index + 1} ----------\n")
                    f.write(f"TIMESTAMP:        {row['timestamp']}\n")
                    f.write(f"SIMILARITY SCORE: {row['hash_similarity']:.4f}\n")
                    f.write(f"\n--- GENERATED REPORT for RANK {index + 1} (from Excel) ---\n")
                    f.write(f"{row['generated_report']}\n")
                    f.write(f"\n--- REAL REPORT for RANK {index + 1} (from RTF) ---\n")
                    f.write(f"{row['real_report']}\n")
                    f.write("-" * (len("---------- RANK 1 ----------")) + "\n")
            print(f"Successfully saved report to '{output_filename}'")
        except Exception as e:
            print(f"Error: Could not write to file '{output_filename}'. Reason: {e}")
    else:
        print("Analysis did not produce any results to save.")


if __name__ == "__main__":
    main()