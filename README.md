# Natural language processing course: `Iterative Refinement of LLM Instructions for Automated Slovenian Traffic News Generation for RTV Slovenija`

## Group members

-   Anže Javornik
-   Timotej Rozina
-   Anže Šavli

## Description

This project addresses the automatic generation of Slovenian traffic news, specifically targeting the needs of RTV Slovenija, where such reports are currently manually prepared. Utilizing Excel data from the `promet.si` portal, the system employs a two-Large Language Model (LLM) approach.

The Slovenian GaMS-9B-Instruct model generates initial traffic reports based on engineered prompts and specific guidelines. Subsequently, the **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)** model evaluates these reports and iteratively refines the instructions provided to GaMS. This methodology focuses on enhancing the quality, accuracy, and adherence of LLM-generated news to RTV Slovenija's stylistic and content requirements through a feedback loop, offering an alternative to direct LLM fine-tuning. The core contribution is an adaptable pipeline demonstrating LLM-driven instruction improvement for specialized Slovenian text generation.

For full details, please refer to the project report: `report.pdf`.

## Methodology

The project followed an iterative approach to develop and refine the automated news generation system:

1.  **Literature Review and Data Exploration**:
    *   Conducted a review of available LLMs, leading to the selection of `cjvt/GaMS-9B-Instruct` for Slovenian text generation and **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)** for evaluation and instruction refinement.
    *   Analyzed and preprocessed the provided traffic data from `promet.si` (Excel format). This involved filtering data by timestamp, consolidating relevant information, and cleaning HTML content.

2.  **Initial Solution: Prompt Engineering**:
    *   Developed an initial comprehensive set of instructions (`default_custom_instructions`) for the GaMS-9B-Instruct model to guide the generation of traffic reports. These instructions covered aspects like urgency-based ordering, sentence structures, naming conventions, and special handling for specific traffic events.

3.  **Evaluation Definition and Implementation**:
    *   Implemented a semi-automatic evaluation mechanism using the **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)** model.
    *   Gemini was prompted with a meta-instruction to:
        *   Assess GaMS's adherence to each specific instruction point.
        *   List instructions followed and not followed.
        *   Provide a numerical adherence score (0-1).
        *   Propose revised instructions for GaMS to improve future outputs.

4.  **LLM Interaction for Iterative Improvement**:
    *   Established an interactive loop orchestrated by `main.py`:
        1.  The user provides a date-time.
        2.  Processed traffic data is fed to GaMS-9B-Instruct with the current set of instructions.
        3.  GaMS's generated output is evaluated by **Gemini 2.5 Flash Preview (`gemini-2.5-flash-preview-04-17`)**, which suggests new/modified instructions.
        4.  The user decides if these new instructions should replace the old ones for subsequent iterations.
    *   This iterative refinement process allows for progressive improvement of GaMS's output without direct model fine-tuning.

5.  **Performance Analysis**:
    *   Assessed the effectiveness of the iterative refinement technique by:
        *   Tracking changes in Gemini's adherence score for GaMS's output across iterations.
        *   Qualitatively analyzing differences in GaMS-generated text, focusing on improvements in areas like urgency ordering, stylistic consistency, and adherence to specific formatting rules.
        *   Implicit human evaluation occurred when the user decided whether Gemini's suggested instruction changes were beneficial.

## Dataset

The primary data sources for this project were:

1.  **Traffic Information Data**:
    *   An Excel file (`Podatki - PrometnoPorocilo_2022_2023_2024.xlsx`) containing traffic information from the `promet.si` portal. This was the direct input for generating traffic news.
2.  **Guidelines and Requirements**:
    *   RTV Slovenija's existing guidelines and requirements for traffic news (implicitly or explicitly derived from documents like `PROMET, osnove.docx`, `PROMET.docx`) informed the development of the instructions for the LLMs and the evaluation criteria.

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
    The script will guide you through providing a date and time to generate traffic news and interact with the instruction refinement process.
