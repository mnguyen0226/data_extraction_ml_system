# Data Extraction ML System

This repository implements a **mock ML system** that extracts structured invoice data from uploaded PDFs.  
The system is implemented as a **Streamlit app** with a focus on **logging, feedback collection, and multi-user support**.  
(No actual ADI/LLM calls are made — data is mocked.)

---

## 📄 Invoice Extraction Flow

1. **Invoice Parsing (Mocked ADI + LLM)**  
   - Given a PDF, the system (mock) extracts:
     - **Invoice-Level (single row)**:  
       - `Invoice Number`, `Total Amount`, `Start Date`, `End Date`.
     - **Item-Level (multi-row)**:  
       - `Fund Tickers`, `Fund Names`, `Fund Amount Allocation`.  
       - Allocation totals must equal the invoice `Total Amount`.

   - The raw output from ADI is two “reasoning” tables, each with:  
     - `attributes`, `values`, `reasoning`, `reference`.

2. **Final Tables**  
   - **Invoice Final Table** → one row of values (attributes → columns).  
   - **Item Final Table** → multiple rows (one per fund).  

---

## 🖥️ Streamlit App UI

- **Authentication**  
  - Users log in with a simple username.  
  - Username is recorded in all logs and feedback.

- **Upload & Process**  
  - Upload a single invoice PDF.  
  - System automatically **identifies the Invoice ID** (mocked).  
  - If invoice is new → process and save all outputs.  
  - If invoice already exists → overwrite the existing 6 CSVs.

- **Reload Processed Invoice**  
  - Users can load any existing invoice from `logs/` to review data and update feedback.  
  - This does **not** re-run extraction — it just loads saved CSVs.

- **AI Reasoning Views**  
  - Show reasoning tables (Arrow-safe).  
  - Includes AI reasoning text + text references.

- **Finalized Outputs**  
  - Invoice-level table (single row).  
  - Item-level table (multi-row).  

- **Human Feedback**  
  - Users can edit `invoice_final` (single row) or `item_final` (multi-row).  
  - Edits include **value changes, row add/remove (for items), and free-text comments**.  
  - Feedback overwrites feedback CSVs.  
  - Stored with `username` and `reviewed_at`.

---

## 📂 Logging Design

Each invoice is stored under a **stable folder**:  

```

logs/{invoice\_id}/
├── invoice\_reasoning.csv
├── item\_reasoning.csv
├── invoice\_final.csv
├── item\_final.csv
├── feedback\_invoice\_final.csv
└── feedback\_item\_final.csv

```

- Exactly **6 CSVs per invoice**.  
- Re-processing an invoice **overwrites all 6 CSVs**.  
- Reloading an invoice just reads from disk.  

### Master Log (`logs/_master_log.csv`)

Append-only CSV with run metadata:

| timestamp | username | invoice_id | pdf_file | invoice_folder | invoice_number | total_amount | items_sum | validation_pass | run_time_sec | run_cost_usd |
|-----------|----------|------------|----------|----------------|----------------|--------------|-----------|-----------------|--------------|--------------|

This allows tracking multiple runs while keeping **only 6 files per invoice folder**.

---

## ⚙️ Pain Points Solved

- ✅ **Stable logging design** → fixed set of 6 files per invoice.  
- ✅ **Human feedback storage** → same schema as final outputs + `Comment`, `username`, `reviewed_at`.  
- ✅ **Multi-user support** → usernames tracked in feedback + master log.  
- ✅ **Run metadata** → runtime and cost logged in master log.  
- ✅ **Reload processed invoices** → edit feedback without rerun.  
- ✅ **Overwrite on rerun** → keeps system simple, ensures canonical data.

---

## 🔄 System Design

- **App**: Streamlit  
- **Storage**: File-based (CSV per invoice in `logs/`)  
- **Data Flow**:  
```

PDF Upload → Mock Extraction → Reasoning Tables → Final Tables
→ Save 4 base CSVs → Collect Human Feedback → Save 2 feedback CSVs
→ Append to master log

````

---

## 🚀 Reproduction

1. Clone this repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```
3. Open in browser:

```
http://localhost:8501
```

---

## 📌 Notes

* No actual ML/LLM inference — all outputs are mocked.
* Designed for **future integration** with ADI + LLM prompting.
* This design emphasizes:

  * Simplicity (only 6 files per invoice).
  * Robustness (feedback always editable).
  * Multi-user logging and audit trail.
