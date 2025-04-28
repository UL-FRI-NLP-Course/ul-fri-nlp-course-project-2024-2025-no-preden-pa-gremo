# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

## Group members

-   Anže Javornik
-   Timotej Rozina
-   Anže Šavli

## Description

LLM, »fine-tune« it, leverage prompt engineering techniques to generate short traffic reports. You are given Excel data from promet.si portal and your goal is to generate regular and important traffic news that are read by the radio presenters at RTV Slovenija. You also need to take into account guidelines and instructions to form the news. Currently, they hire students to manually check and type reports that are read every 30 minutes.

## Methodology

1. Literature Review: Conduct a thorough review of existing research and select appropriate LLMs for the task. Review and prepare an exploratory report on the data provided.

2. Initial solution: Try to solve the task initially only by using prompt engineering techniques.

3. Evaulation definition: Define (semi-)automatic evaluation criteria and implement it. Take the following into account: identification of important news, correct roads namings, correct filtering, text lengths and words, ...

4. LLM (Parameter-efficient) fine-tuning: Improve an existing LLM to perform the task automatically. Provide an interface to do an interactive test.

5. Evaluation and Performance Analysis: Assess the effectiveness of each technique by measuring improvements in model performance, using appropriate automatic (P, R, F1) and human evaluation metrics.

## Dataset

1. RTV Slo data:

    The data consists of:

    - Promet.si input resources (Podatki - PrometnoPorocilo_2022_2023_2024.xlsx).

    - RTV Slo news texts to be read through the radio stations (Podatki - rtvslo.si).

    - Additional instructions for the students that manually type news texts (PROMET, osnove.docx, PROMET.docx).

## Submission 2 changes

1. Implemented functionality

    - Inside LLMs folder we implemented two different LLMs (gaMS and Gemini).
    - The data is being processed in readData file located in Data folder. It takes data from 30 minutes interval and calculates the similarity between the columns data (A1, B1, C1, A2, B2, C2).
    - The file returns prompt text that is parsed into selected LLM model.

2. Result analysis

    - The LLMs correctly parse all the given news and generates text.
    - Both LLMs need to be tunned to follow importance rules / sequence. In this regard Gemini currently stongly outperforms gaMS.
    - Both LLMs have some problems with formatting slovene statements.

3. Future

    - We need to implement better learning of rules for both models.
    - We need to implement "Odpoved" functionality
    - We need to implement better statement formatting and upgrade analysis algorithms to check the statements and fix them

4. Current results examples:

    - Gemini

    ```
    Poslušate prometne informacije.

    Podatki o prometu:

    1.  Na Primorski avtocesti proti Ljubljani ovira promet okvarjeno vozilo v predoru Kastelec. Zaprt je vozni pas.
    2.  Pričakujemo daljše čakalne dobe na vstopnih točkah iz Avstrije v Slovenijo. Pred predorom Karavanke na Gorenjski avtocesti proti Karavankam in na mejnem prehodu Šentilj od Maribora proti Šentilju. Glavni razlog je nakup vinjet na meji. Voznikom priporočamo, da vinjeto kupijo prek spleta.
    3.  V Ljubljani je zaradi del na cesti med 4. in 7. januarjem predvidena popolna zapora Dunajske ceste. Obvoz je urejen po Samovi, Drenikovi, Celovški in Tivolski cesti.
    4.  Zaradi praznika danes od 8. do 22. ure velja prepoved prometa tovornih vozil, katerih največja dovoljena masa presega 7,5 ton.
    5.  Ponekod po Sloveniji megla v pasovih zmanjšuje vidljivost.
    6.  Na cesti čez prelaz Vršič so obvezne verige.

    Želimo vam srečno in varno pot.
    ```

    - gaMS

    ```
    Zaradi praznika velja omejitev prometa tovornih vozil, katerih največja dovoljena masa presega 7,5 t, danes od 8. do 22. ure. Ponekod po Sloveniji megla v pasovih zmanjšuje vidljivost.

    Na primorski avtocesti v predoru Kastelec proti Ljubljani ovire zaradi okvarjenega vozila. Zaprt je vozni pas. Obvezne so verige na cesti čez prelaz Vršič.

    V Ljubljani je med 4. in 7. 1. predvidena popolna zapora Dunajske ceste. Obvoz bo urejen po Samovi, Drenikovi, Celovški in Tivolski cesti. Več o delovnih zaporah v prometni napovedi.

    Želimo vam srečno in varno na cestah v letu 2024! As of January 1 neighbouring country Croatia has entered the Schengen Area. Border control between Slovenia and Croatia is lifted. There is no need to stop at borders. Drive carefully! Buy vignette for Slovenia online.

    Thererore long queues are expected in entering points from Austria to Slovenia, i.e. Karavanke tunnel (A2) and Sentilj/Spielfeld crossing (A1). Important reason for these queues is that drivers don't have vignette for Slovenian roads and have to buy them at the border. To reduce or even avoid long waiting periods drivers are strongly recommended to buy vignette for Slovenian motorways online. They can do it here.
    ```
