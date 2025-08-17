# Data Extraction ML System
Mock ML system that extract data given the PDF that parse the invoice data to get invoice-level and invoice-item. With the following steps:
- Using Azure Document Intelligence (ADI) + LLM prompting to parse invoice data thus we got invoice-level and invoice-item. Invoice-level includes: Invoice Number, Total Amount, Start Date, End Date. Invoice-item includes: Fund Tickers, Fund Names, Fund Amount Allocation. Note that the total of Fund Amount Allocation is equal to Total Amount. ADI will return 2 dataframes (invoice-level and invoice-item) with each value is a row. Each table will have the following columns: attributes, values, reasoning (AI extraction reasoning), reference (text reference).
- The system will then return the 2 dataframes: invoice-level and item-level that contain only the attributes and values. Each table will only has 1 rows with each row is the values of each attribute column.

The system is a Streamlit app will have the following components:
- PDF uploading components that will save down the PDF.
- A container that st.success the steps above.
- A container for "AI Reasoning - Invoice-Level" that show the dataframe extracted by ADI.
- A container for "AI Reasoning - Invoice-Item" that show the dataframe extracted by ADI.
- st.columns(2): One half for showing the uploaded PDF, One half for showing the 2 finalized dataframes with attributes and values.

Notes: There will be no ADI model or LLMs. We can just use mock data initially

Pain point:
- How do we design the logging system?
- How should we design human feedback to review final for later changing the LLMs training
    - [X] Logging such as allow to change values in Invoice-Level but for Items-level, not only change but can add or delete rows
    - [X] There should be comments too how should we design this for storage. User can approve too -> How to desgin this to make it user friendly.
    - Global: Feature request to button to a share Excel link
    - [X] What about logging information like running cost, running time
    - This app will be used by multiple user. Each invoice should only has the 4 copy of invoie_final, invoie_reason, item_final, item_reasoning. User logging, Feedback should be
    - Invoice reload?
    - Enable delete and reload