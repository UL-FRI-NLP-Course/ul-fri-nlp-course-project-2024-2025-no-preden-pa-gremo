import pandas as pd
from datetime import datetime, timedelta
import Levenshtein
from bs4 import BeautifulSoup

data_file = "./RTVSlo/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"

try:
    df = pd.read_excel(data_file, sheet_name="2024")

    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    input_time = datetime.strptime("2024-01-02 08:17:07", "%Y-%m-%d %H:%M:%S")  # Replace with your input time

    start_time = input_time - timedelta(minutes=30)

    mask = (df['Datum'] >= start_time) & (df['Datum'] <= input_time)
    filtered_df = df.loc[mask]

    print(filtered_df)

    # Filter out columns where all values are NaN
    filtered_df = filtered_df.dropna(axis=1, how='all')

    # Print the filtered DataFrame
    print("\nFiltered DataFrame without columns containing all NaN values:")
    print(filtered_df)

    columns_to_combine = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2']
    columns_to_combine = [col for col in columns_to_combine if col in filtered_df.columns]

    filtered_df['Combined'] = filtered_df[columns_to_combine].apply(
        lambda row: ' '.join(row.dropna().astype(str)), axis=1
    )


        # Print the resulting DataFrame
    print("\nFiltered DataFrame with combined columns:")
    print(filtered_df[['Datum', 'Combined']])  # Display only the relevant columns

    combined_values = filtered_df['Combined'].tolist()
    similarity_matrix = []
    threshold = 0.9

    for i in range(len(combined_values)):
        row_similarities = []
        for j in range(len(combined_values)):
            # Compute similarity using Levenshtein ratio
            similarity = Levenshtein.ratio(combined_values[i], combined_values[j])
            row_similarities.append(similarity)
        similarity_matrix.append(row_similarities)

    # Print the similarity matrix
    print("\nSimilarity Matrix:")
    for row in similarity_matrix:
        print(row)


    all_above_threshold = all(
            similarity >= threshold for row in similarity_matrix for similarity in row if row != similarity_matrix[0]
        )

    if all_above_threshold:
        # Find the row with the highest similarity sum
        similarity_sums = [sum(row) for row in similarity_matrix]
        max_similarity_index = similarity_sums.index(max(similarity_sums))

        # Select the 'Combined' field from the row with the highest similarity
        selected_combined = combined_values[max_similarity_index]
        print(f"\nSelected Combined Field: {selected_combined} \n\n")
        print(BeautifulSoup(selected_combined, "html.parser").get_text())
    else:
        print("\nNot all similarities are above the threshold.")

except Exception as e:
    print(f"Error reading {data_file}: {e}")
    df = None  # Set to None if loading fails