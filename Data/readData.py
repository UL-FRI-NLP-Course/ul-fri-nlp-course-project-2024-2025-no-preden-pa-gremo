import pandas as pd
from datetime import datetime, timedelta
import Levenshtein
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import locale, re
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
from pathlib import Path

data_file = "./Data/RTVSlo/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"

def get_final_traffic_text(input_time_str, threshold=0.8):
    try:
        t_start = datetime.strptime(input_time_str, "%Y-%m-%d %H:%M:%S")

        df = pd.read_excel(data_file, sheet_name=f"{t_start.year}")

        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        input_time = datetime.strptime(input_time_str, "%Y-%m-%d %H:%M:%S")
        start_time = input_time - timedelta(minutes=5)
        mask = (df['Datum'] >= start_time) & (df['Datum'] <= input_time)
        filtered_df = df.loc[mask]
        # Filter out columns where all values are NaN
        filtered_df = filtered_df.dropna(axis=1, how='all')

        columns1 = [col for col in ['A1', 'B1', 'C1'] if col in filtered_df.columns]
        columns2 = [col for col in ['A2', 'B2', 'C2'] if col in filtered_df.columns]

        filtered_df['Combined1'] = filtered_df[columns1].apply(
            lambda row: ' '.join(row.dropna().astype(str)), axis=1
        )
        filtered_df['Combined2'] = filtered_df[columns2].apply(
            lambda row: ' '.join(row.dropna().astype(str)), axis=1
        )

        combined_values = filtered_df['Combined1'].tolist()
        similarity_matrix = []

        for i in range(len(combined_values)):
            row_similarities = []
            for j in range(len(combined_values)):
                similarity = Levenshtein.ratio(combined_values[i], combined_values[j])
                row_similarities.append(similarity)
            similarity_matrix.append(row_similarities)

        all_above_threshold = all(
            similarity >= threshold for row in similarity_matrix for similarity in row if row != similarity_matrix[0]
        )

        if combined_values:
            if all_above_threshold:
                # Use the most similar row
                similarity_sums = [sum(row) for row in similarity_matrix]
                max_similarity_index = similarity_sums.index(max(similarity_sums))
                selected_combined = combined_values[max_similarity_index]
            else:
                # Use the latest line
                selected_combined = combined_values[-1]

            kept_lines = []

            soup = BeautifulSoup(selected_combined, "html.parser")
            for p in soup.find_all("p"):
                if p.find("a"):
                    p.decompose()
                    continue
                strong = p.find("strong")
                if strong and strong.string:
                    word_count = len(strong.string.strip().split())
                    if word_count <= 3:
                        strong.decompose()
                        if not p.get_text(strip=True):
                            p.decompose()
                        continue
                text = p.get_text(strip=True)
                if text:
                    kept_lines.append(text)

            processed_lines = []
            for line in kept_lines:
                line = line.strip()
                if line:
                    if line[-1] not in {'.', ',', '?', '!'}:
                        line += '.'
                    line = line[0].upper() + line[1:] if line else line
                    processed_lines.append(line)

            final_text = '\n'.join(processed_lines)
            return final_text
        else:
            return None

    except Exception as e:
        print(f"Error reading {data_file}: {e}")
        return None

MARKER = "Podatki o prometu."

def get_real_traffic_report(input_time_str: str):
    t_start = datetime.strptime(input_time_str, "%Y-%m-%d %H:%M:%S")
    t_end   = t_start + timedelta(minutes=15)

    # Slovene month name
    try:
        locale.setlocale(locale.LC_TIME, "sl_SI.UTF-8")
        month_sl = t_start.strftime("%B").capitalize()
    except locale.Error:
        month_sl = ["Januar","Februar","Marec","April","Maj","Junij",
                    "Julij","Avgust","September","Oktober","November","December"][t_start.month-1]

    dir_path = Path(f"./Data/RTVSlo/Podatki - rtvslo.si/Promet {t_start.year}/{month_sl} {t_start.year}")
    if not dir_path.is_dir():
        raise FileNotFoundError(dir_path.resolve())

    stamp_rx = re.compile(
        r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\s+[\t ]+\s*(\d{1,2})\.(\d{2})"
    )

    for rtf_file in sorted(dir_path.glob("*.rtf")):
        try:
            with rtf_file.open("r", encoding="utf-8", errors="ignore") as f:
                raw = rtf_to_text(f.read())
        except Exception:
            continue

        # Is this file in the time window?
        for m in stamp_rx.finditer(raw):
            d, mth, y, h, mi = map(int, m.groups())
            try:
                stamp = datetime(y, mth, d, h, mi)
            except ValueError:
                continue
            if t_start <= stamp <= t_end:
                # ── TRIM HERE ────────────────────────────────────────────
                body = raw.split(MARKER, 1)[-1]           # everything after header
                body = body.lstrip()                      # drop leading newlines/spaces
                body = "\n".join(ln for ln in body.splitlines() if ln.strip())  # keep only non-empty lines
                print(f"Found traffic report in {rtf_file.name} at {stamp}")
                return body

    return None


def preload_real_reports(start_date: datetime, end_date: datetime):
    """
    Scans all RTF files within a date range and loads them into a dictionary
    mapping timestamps to report content. This avoids repeated file access.
    """
    print("Pre-loading all real reports from RTF files...")
    report_cache = {}

    # Slovene month name setup
    try:
        locale.setlocale(locale.LC_TIME, "sl_SI.UTF-8")
    except locale.Error:
        print("Slovene locale not found, using manual month names.")

    stamp_rx = re.compile(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\s+[\t ]+\s*(\d{1,2})\.(\d{2})")
    MARKER = "Podatki o prometu."

    # Iterate through the years and months in the requested date range
    for year in range(start_date.year, end_date.year + 1):
        start_month = start_date.month if year == start_date.year else 1
        end_month = end_date.month if year == end_date.year else 12
        for month in range(start_month, end_month + 1):

            current_dt = datetime(year, month, 1)
            try:
                month_sl = current_dt.strftime("%B").capitalize()
            except ValueError:  # Handle manual months if locale fails
                month_sl = ["Januar", "Februar", "Marec", "April", "Maj", "Junij",
                            "Julij", "Avgust", "September", "Oktober", "November", "December"][month - 1]

            dir_path = Path(f"./Data/RTVSlo/Podatki - rtvslo.si/Promet {year}/{month_sl} {year}")
            if not dir_path.is_dir():
                continue

            for rtf_file in tqdm(sorted(dir_path.glob("*.rtf")), desc=f"Scanning {dir_path.name}", leave=False):
                try:
                    with rtf_file.open("r", encoding="utf-8", errors="ignore") as f:
                        raw = rtf_to_text(f.read())
                except Exception:
                    continue

                # Find all timestamps in the file
                for m in stamp_rx.finditer(raw):
                    d, mth, y, h, mi = map(int, m.groups())
                    try:
                        stamp = datetime(y, mth, d, h, mi)
                    except ValueError:
                        continue

                    # Extract the report body and cache it
                    body = raw.split(MARKER, 1)[-1].strip()
                    body = "\n".join(ln for ln in body.splitlines() if ln.strip())
                    report_cache[stamp] = body

    print(f"Finished pre-loading. Found {len(report_cache)} real reports.")
    return report_cache


def preload_generated_reports(data_file: str):
    """
    Reads the Excel data and pre-generates all possible traffic reports.
    It now uses a cache file to avoid re-processing on subsequent runs.
    """
    # Define a path for our cache file
    cache_file = Path("./generated_reports_cache.pkl")

    # 1. CHECK IF THE CACHE FILE EXISTS
    if cache_file.is_file():
        print(f"Loading pre-generated reports from cache: '{cache_file}'")
        try:
            with open(cache_file, 'rb') as f:
                report_cache = pickle.load(f)
            print(f"Successfully loaded {len(report_cache)} reports from cache.")
            return report_cache
        except Exception as e:
            print(f"Warning: Could not load cache file. Re-generating. Error: {e}")

    # 2. IF CACHE DOES NOT EXIST, RUN THE ORIGINAL SLOW PROCESS
    print("No cache found. Pre-generating all reports from Excel data (this will take a few minutes)...")
    report_cache = {}

    # Load the entire dataset from all sheets
    all_dfs = [pd.read_excel(data_file, sheet_name=str(year), usecols=['Datum', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2']) for
               year in range(2022, 2025)]
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Datum'] = pd.to_datetime(full_df['Datum'], errors='coerce').dt.floor('T')
    full_df.dropna(subset=['Datum'], inplace=True)

    grouped = full_df.groupby('Datum')

    for timestamp, group_df in tqdm(grouped, desc="Generating reports from Excel"):
        input_time_str = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        columns1 = [col for col in ['A1', 'B1', 'C1'] if col in group_df.columns]
        group_df['Combined1'] = group_df[columns1].apply(
            lambda row: ' '.join(row.dropna().astype(str)), axis=1
        )

        if group_df['Combined1'].empty:
            continue

        selected_combined = group_df['Combined1'].iloc[-1]
        soup = BeautifulSoup(selected_combined, "html.parser")
        lines = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

        processed_lines = []
        for line in lines:
            if line and line[-1] not in {'.', ',', '?', '!'}:
                line += '.'
            processed_lines.append(line[0].upper() + line[1:])

        final_text = '\n'.join(processed_lines)
        if final_text:
            report_cache[pd.to_datetime(timestamp)] = final_text

    # 3. SAVE THE NEWLY GENERATED DATA TO THE CACHE FILE FOR NEXT TIME
    print(f"Finished generating {len(report_cache)} reports. Saving to cache file...")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(report_cache, f)
        print(f"Successfully saved cache to '{cache_file}'")
    except Exception as e:
        print(f"Error: Could not save cache file. Reason: {e}")

    return report_cache


def analyze_reports(start_date_str: str, end_date_str: str):
    """
    Performs an ultra-fast analysis using full pre-loading of both generated
    and real reports, with a fast HashingVectorizer for similarity.
    NOW INCLUDES DE-DUPLICATION LOGIC.
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

    # 1. PRE-LOAD (no changes here)
    real_report_cache = preload_real_reports(start_dt, end_dt)
    generated_report_cache = preload_generated_reports(data_file)

    # 2. MATCH (no changes here)
    print("Matching cached reports...")
    results_list = []

    for gen_ts, gen_report in tqdm(generated_report_cache.items(), desc="Matching reports"):
        if not (start_dt <= gen_ts <= end_dt):
            continue
        for real_ts, real_report in real_report_cache.items():
            if gen_ts <= real_ts <= gen_ts + timedelta(minutes=15):
                results_list.append({
                    'timestamp': gen_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    'generated_report': gen_report,
                    'real_report': real_report
                })
                break

    if not results_list:
        print("No pairs of generated and real reports could be matched.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)

    # 3. SIMILARITY CALCULATION (no changes here)
    print(f"Calculating similarity for {len(results_df)} report pairs using HashingVectorizer...")
    vectorizer = HashingVectorizer(stop_words='english', n_features=2 ** 18)
    generated_vectors = vectorizer.transform(results_df['generated_report'])
    real_vectors = vectorizer.transform(results_df['real_report'])
    similarities = (generated_vectors * real_vectors.T).diagonal()
    results_df['hash_similarity'] = similarities

    # ------------------------------------------------------------------
    # --- NEW: 4. FILTER FOR UNIQUE BEST MATCHES ---
    # ------------------------------------------------------------------
    print(f"\nOriginal matched pairs found: {len(results_df)}")

    # First, sort by similarity score, highest to lowest.
    # This ensures that the first entry for any given real_report is its best match.
    results_df = results_df.sort_values(by='hash_similarity', ascending=False)

    # Now, drop duplicates based on the 'real_report' column, keeping only the first occurrence.
    results_df = results_df.drop_duplicates(subset=['real_report'], keep='first')

    print(f"Filtered down to {len(results_df)} unique best matches.")
    # ------------------------------------------------------------------

    print("Analysis complete.")
    # The final sorting for the report will be handled in the main block.
    return results_df