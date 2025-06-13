# Natural language processing course: `Comparative Analysis of LLMs in an Iterative Refinement Pipeline for Automated Slovenian Traffic News Generation for RTV Slovenija`

## Group members

-   Anže Javornik
-   Timotej Rozina
-   Anže Šavli

## Description

This project develops and evaluates an automated pipeline for generating Slovenian traffic news, specifically targeting the needs of RTV Slovenija, where such reports are currently manually prepared. The system employs a multi-Large Language Model (LLM) approach to compare different generation models and iteratively refine their instructions.

Using Excel data from the `promet.si` portal, matched against historical ground-truth reports from RTV Slovenija, a generator model—either the Slovenian-specific **GaMS-9B-Instruct** or the multilingual **Gemma-7B-IT**—produces an initial traffic report. A more powerful evaluation model, **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)**, then assesses this output against specific guidelines and proposes improvements to the instructions for the next iteration.

This feedback loop, which serves as an alternative to direct model fine-tuning, was tested on a pre-selected dataset of 10 real-world traffic scenarios. Performance is measured quantitatively with BLEU and BERTScore against ground-truth reports, and qualitatively through Gemini's structured evaluation. The project's core contribution is a comparative analysis of the two generator models within this adaptive pipeline, demonstrating a robust method for improving specialized Slovenian text generation through automated instruction refinement.

For full details, including comprehensive results and logs, please refer to the project report: `report.pdf`.

## Methodology

The project followed an iterative approach to develop and refine the automated news generation system, focusing on comparative analysis and a robust evaluation-refinement loop:

1.  **Literature Review and Model Selection**:
    *   Conducted a review of available LLMs, leading to the selection of `cjvt/GaMS-9B-Instruct` for its Slovenian-specific training, Google's `Gemma-7B-IT` as a multilingual baseline for comparison, and **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)** for its advanced reasoning capabilities in evaluation and instruction refinement.

2.  **Data Preparation and Testbed Creation**:
    *   Developed the `analyze_reports` function to process historical `promet.si` Excel data and compare it with corresponding ground-truth RTV Slovenija reports.
    *   Identified the top 10 historical instances where input data and ground-truth output were most closely aligned to create a consistent and reproducible testbed for all experiments.
    *   The `get_final_traffic_text` function in `Data/readData.py` handles the extraction and cleaning of traffic data for specific timestamps.

3.  **Models and Initial Instructions**:
    *   Three LLMs were integrated: GaMS-9B-Instruct (generator), Gemma-7B-IT (generator), and Gemini 2.5 Flash Preview (evaluator/refiner).
    *   An initial set of comprehensive `default_custom_instructions` was developed to provide a strong baseline for guiding the generator models, covering criteria like urgency-based ordering, sentence structures, naming conventions, and specific event handling rules.

4.  **Automated Iterative Refinement Loop**:
    *   The core pipeline, managed by `main.py`, executes a five-step iterative refinement process for each of the 10 test cases, for both GaMS and Gemma models.
    *   This loop consists of:
        *   **Generation**: The chosen generator model produces a news report based on current instructions and input data.
        *   **Quantitative Evaluation**: The generated report is automatically scored against its ground-truth counterpart using BLEU Score (n-gram precision/fluency) and BERTScore (semantic similarity - Precision, Recall, F1-Score).
        *   **Qualitative Evaluation & Refinement**: Gemini receives the generated report and its instructions, providing qualitative scores (1-5) on Grammar (`Slovnica`), Hierarchy (`Hierarhija`), Composition (`Sestava`), Naming (`Poimenovanje`), and General adherence (`Generalna`). It also proposes revised instructions.
        *   **Iteration**: Gemini's refined instructions automatically replace the previous set for the next run.

5.  **Performance Analysis**:
    *   Assessed the effectiveness by comparing the average scores and score progression (over five iterations) for both GaMS and Gemma.
    *   Analyzed qualitative scores from Gemini to identify strengths and weaknesses.
    *   Conducted a separate consistency check of Gemini 2.5 Flash's evaluation abilities by running it 10 times on the same input, providing insights into evaluation reliability, particularly for subjective criteria like "Sestava".

## Dataset

The project utilized a custom dataset derived from two primary sources to enable robust evaluation:

1.  **Traffic Information Data**:
    *   An Excel file (`Podatki - PrometnoPorocilo_2022_2023_2024.xlsx`) containing raw traffic information from the `promet.si` portal. This data served as the input for LLM generation.

2.  **Historical Ground-Truth Reports**:
    *   Corresponding final reports published by RTV Slovenija, matched with the raw traffic data using a similarity metric. These ground-truth reports served as the reference for quantitative (BLEU, BERTScore) evaluation.

From these, a standardized testbed of 10 real-world traffic scenarios was carefully selected, ensuring consistent input and a verifiable ground truth for all comparative experiments.

## How to Run

1.  **Prerequisites**:
    *   Python 3.x
    *   pip

2.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file with the content provided (see list of dependencies in the prompt) and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google API Key**:
    *   You will need a Google API key for the Gemini model.
    *   Open the file `LLMs/gemy.py`.
    *   Locate the line `genai.configure(api_key="")`.
    *   Replace the placeholder or empty string with your valid Google API key.
    *   Alternatively, modify the script to load the API key from an environment variable for better security.

5.  **Run the Project**:
    Execute the main script from the project's root directory:
    ```bash
    python main.py
    ```
    The script will guide you through providing a date and time to generate traffic news and interact with the instruction refinement process, allowing you to observe the comparative performance of the models.
