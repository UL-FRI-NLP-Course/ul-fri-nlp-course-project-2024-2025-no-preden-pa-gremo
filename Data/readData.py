import pandas as pd
from datetime import datetime, timedelta
import Levenshtein
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import locale, re
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


print(get_real_traffic_report("2023-04-19 18:35:00"))
result = get_final_traffic_text("2023-04-19 18:35:00")
print(result)