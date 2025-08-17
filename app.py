# app.py
import streamlit as st
import pandas as pd
import os
import hashlib
from datetime import date, timedelta
import datetime
import time

st.set_page_config(page_title="Data Extraction ML System — Mock", layout="wide")

# -----------------------------
# Paths & setup
# -----------------------------
UPLOAD_DIR = "uploads"
LOG_ROOT = "logs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)

# Exactly 6 CSVs per invoice folder
BASE_FILES = [
    "invoice_reasoning.csv",
    "item_reasoning.csv",
    "invoice_final.csv",
    "item_final.csv",
]
FEEDBACK_FILES = [
    "feedback_invoice_final.csv",
    "feedback_item_final.csv",
]


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _file_hash(bytes_obj: bytes) -> int:
    # Seed for mock extraction
    return int(hashlib.md5(bytes_obj).hexdigest(), 16)


@st.cache_data(show_spinner=False)
def save_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    safe_name = uploaded_file.name.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{digest}__{safe_name}")
    with open(save_path, "wb") as out:
        out.write(file_bytes)
    return save_path


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type columns to string for Streamlit/Arrow display."""
    safe = df.copy()
    for col in safe.columns:
        try:
            if safe[col].map(type).nunique() > 1:
                safe[col] = safe[col].astype(str)
        except Exception:
            safe[col] = safe[col].astype(str)
    return safe


def invoice_folder(invoice_id: str) -> str:
    f = os.path.join(LOG_ROOT, invoice_id)
    os.makedirs(f, exist_ok=True)
    return f


def _write_atomic(df: pd.DataFrame, target_path: str):
    tmp = target_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, target_path)


def list_existing_invoice_ids() -> list[str]:
    if not os.path.isdir(LOG_ROOT):
        return []
    ids = []
    for name in sorted(os.listdir(LOG_ROOT)):
        p = os.path.join(LOG_ROOT, name)
        if os.path.isdir(p) and all(
            os.path.exists(os.path.join(p, fn)) for fn in BASE_FILES
        ):
            ids.append(name)
    return ids


# -----------------------------
# Mock ADI + LLM Extraction
# -----------------------------
def _mock_invoice_payload(seed: int):
    fund_catalog = [
        ("BLKX", "BlackRock Core Bond Fund"),
        ("VFINX", "Vanguard 500 Index"),
        ("FXAIX", "Fidelity 500 Index"),
        ("IWM", "iShares Russell 2000 ETF"),
        ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
        ("QQQ", "Invesco QQQ Trust"),
    ]
    n = 3 + (seed % 3)  # 3–5 items
    start_ix = seed % (len(fund_catalog) - n + 1)
    items = fund_catalog[start_ix : start_ix + n]

    base = 1000 * (1 + (seed % 4))  # 1k–4k
    amounts = [base * (i + 1) for i in range(n)]
    total_amount = sum(amounts)

    start_dt = date(2025, 1 + (seed % 6), 1 + (seed % 20))
    end_dt = start_dt + timedelta(days=29)
    invoice_number = f"INV-{(seed % 900000) + 100000}"

    return {
        "invoice": {
            "Invoice Number": invoice_number,
            "Total Amount": float(total_amount),
            "Start Date": start_dt.strftime("%Y-%m-%d"),
            "End Date": end_dt.strftime("%Y-%m-%d"),
        },
        "items": [
            {"Fund Tickers": t, "Fund Names": n, "Fund Amount Allocation": float(a)}
            for (t, n), a in zip(items, amounts)
        ],
    }


def _seed_from_pdf(pdf_path: str) -> int:
    with open(pdf_path, "rb") as f:
        return _file_hash(f.read())


@st.cache_data(show_spinner=False)
def mock_adi_extract_from_seed(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return reasoning dataframes and invoice_id (== invoice number)."""
    payload = _mock_invoice_payload(seed)

    inv_rows = []
    for attr, val in payload["invoice"].items():
        inv_rows.append(
            {
                "attributes": attr,
                "values": val,
                "reasoning": f"Located near '{attr}' label via header proximity heuristic.",
                "reference": "Page 1: lines 8–15",
            }
        )
    invoice_level_df = pd.DataFrame(
        inv_rows, columns=["attributes", "values", "reasoning", "reference"]
    )

    item_rows = []
    for i, item in enumerate(payload["items"], start=1):
        for attr in ["Fund Tickers", "Fund Names", "Fund Amount Allocation"]:
            item_rows.append(
                {
                    "attributes": attr,
                    "values": item[attr],
                    "reasoning": f"Extracted from line-item {i} table cell using cell-boundary cues.",
                    "reference": f"Page 1: table row {i}",
                }
            )
    item_level_df = pd.DataFrame(
        item_rows, columns=["attributes", "values", "reasoning", "reference"]
    )

    invoice_id = payload["invoice"]["Invoice Number"]  # Auto-identified (mock)
    return invoice_level_df, item_level_df, invoice_id


# -----------------------------
# Post-processing to final tables
# -----------------------------
@st.cache_data(show_spinner=False)
def to_final_invoice_table(invoice_reasoning_df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        invoice_reasoning_df[["attributes", "values"]]
        .set_index("attributes")
        .T.reset_index(drop=True)
    )
    return wide


@st.cache_data(show_spinner=False)
def to_final_items_table(item_reasoning_df: pd.DataFrame) -> pd.DataFrame:
    grouped = item_reasoning_df.groupby("attributes")["values"].apply(list).to_dict()
    tickers = grouped.get("Fund Tickers", []) or []
    names = grouped.get("Fund Names", []) or []
    amts = pd.to_numeric(
        pd.Series(grouped.get("Fund Amount Allocation", []) or []), errors="coerce"
    ).tolist()
    m = max(len(tickers), len(names), len(amts), 1)

    def _pad(lst, fill, n):
        return (lst + [fill] * (n - len(lst)))[:n]

    tickers = _pad(tickers, "", m)
    names = _pad(names, "", m)
    amts = _pad(amts, 0.0, m)
    return pd.DataFrame(
        {
            "Fund Tickers": tickers,
            "Fund Names": names,
            "Fund Amount Allocation": amts,
        }
    )


@st.cache_data(show_spinner=False)
def validate_totals(
    invoice_final_df: pd.DataFrame, items_reasoning_df: pd.DataFrame
) -> tuple[bool, float, float]:
    try:
        total = float(invoice_final_df.loc[0, "Total Amount"])
    except Exception:
        total = pd.to_numeric(invoice_final_df.loc[0, "Total Amount"], errors="coerce")
    raw_allocs = items_reasoning_df[
        items_reasoning_df["attributes"] == "Fund Amount Allocation"
    ]["values"].tolist()
    alloc_total = float(pd.to_numeric(pd.Series(raw_allocs), errors="coerce").sum())
    return (round(alloc_total, 2) == round(float(total), 2), alloc_total, float(total))


# -----------------------------
# Logging (append-only master; fixed per-invoice folder)
# -----------------------------
def log_overwrite_base_csvs(
    folder: str,
    invoice_reasoning: pd.DataFrame,
    item_reasoning: pd.DataFrame,
    invoice_final: pd.DataFrame,
    item_final: pd.DataFrame,
):
    _write_atomic(invoice_reasoning, os.path.join(folder, "invoice_reasoning.csv"))
    _write_atomic(item_reasoning, os.path.join(folder, "item_reasoning.csv"))
    _write_atomic(invoice_final, os.path.join(folder, "invoice_final.csv"))
    _write_atomic(item_final, os.path.join(folder, "item_final.csv"))


def init_or_reset_feedback_csvs(folder: str, username: str):
    """Create/overwrite empty feedback CSVs with headers only."""
    inv_fb_path = os.path.join(folder, "feedback_invoice_final.csv")
    item_fb_path = os.path.join(folder, "feedback_item_final.csv")

    # Try to scaffold invoice feedback columns from latest invoice_final
    try:
        inv_final = pd.read_csv(os.path.join(folder, "invoice_final.csv"))
        inv_cols = list(inv_final.columns)
    except Exception:
        inv_cols = ["Invoice Number", "Total Amount", "Start Date", "End Date"]

    invoice_fb_empty = pd.DataFrame(
        columns=inv_cols + ["Comment", "reviewed_at", "username"]
    )
    _write_atomic(invoice_fb_empty, inv_fb_path)

    item_fb_empty = pd.DataFrame(
        columns=[
            "row_index",
            "Fund Tickers",
            "Fund Names",
            "Fund Amount Allocation",
            "Comment",
            "reviewed_at",
            "username",
        ]
    )
    _write_atomic(item_fb_empty, item_fb_path)


def append_master_log(
    username: str,
    invoice_id: str,
    pdf_path: str | None,
    invoice_final: pd.DataFrame,
    validation: tuple[bool, float, float],
    run_time_sec: float,
    run_cost_usd: float,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ok, alloc_sum, total_amt = validation
    master_path = os.path.join(LOG_ROOT, "_master_log.csv")
    row = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "username": username or "",
                "invoice_id": invoice_id,
                "pdf_file": pdf_path or "",
                "invoice_folder": os.path.join(LOG_ROOT, invoice_id),
                "invoice_number": (
                    invoice_final.get("Invoice Number", [""])[0]
                    if not invoice_final.empty
                    else ""
                ),
                "total_amount": float(total_amt) if pd.notna(total_amt) else "",
                "items_sum": float(alloc_sum) if pd.notna(alloc_sum) else "",
                "validation_pass": bool(ok),
                "run_time_sec": float(run_time_sec),
                "run_cost_usd": float(run_cost_usd),
            }
        ]
    )
    if os.path.exists(master_path):
        row.to_csv(master_path, mode="a", header=False, index=False)
    else:
        row.to_csv(master_path, index=False)


# -----------------------------
# Feedback saving (SAME schema + Comment + Username)
# -----------------------------
def save_feedback_invoice(folder: str, edited_wide_df: pd.DataFrame, username: str):
    out = edited_wide_df.copy()
    out["reviewed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out["username"] = username
    _write_atomic(out, os.path.join(folder, "feedback_invoice_final.csv"))


def save_feedback_items(folder: str, edited_items_df: pd.DataFrame, username: str):
    for c in ["Fund Tickers", "Fund Names", "Fund Amount Allocation", "Comment"]:
        if c not in edited_items_df.columns:
            edited_items_df[c] = "" if c != "Fund Amount Allocation" else 0.0
    out = edited_items_df[
        ["Fund Tickers", "Fund Names", "Fund Amount Allocation", "Comment"]
    ].copy()
    id_mask = (
        out[["Fund Tickers", "Fund Names"]]
        .fillna("")
        .apply(lambda s: s.str.strip() != "", axis=0)
        .any(axis=1)
    )
    amt_mask = pd.to_numeric(out["Fund Amount Allocation"], errors="coerce").notna()
    out = out[id_mask | amt_mask]
    out["Fund Amount Allocation"] = pd.to_numeric(
        out["Fund Amount Allocation"], errors="coerce"
    ).fillna(0.0)
    out.insert(0, "row_index", range(1, len(out) + 1))
    out["reviewed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    out["username"] = username
    _write_atomic(out, os.path.join(folder, "feedback_item_final.csv"))


# -----------------------------
# Load feedback (prefer existing feedback; fallback to finals)
# -----------------------------
def load_feedback_invoice_or_default(
    folder: str, inv_final_df: pd.DataFrame
) -> pd.DataFrame:
    p = os.path.join(folder, "feedback_invoice_final.csv")
    try:
        if os.path.exists(p):
            fb = pd.read_csv(p)
            drop_cols = [c for c in ["username", "reviewed_at"] if c in fb.columns]
            fb_view = fb.drop(columns=drop_cols, errors="ignore")
            if not fb_view.empty:
                return fb_view
    except Exception:
        pass
    df = inv_final_df.copy()
    if "Comment" not in df.columns:
        df["Comment"] = ""
    return df


def load_feedback_items_or_default(
    folder: str, item_final_df: pd.DataFrame
) -> pd.DataFrame:
    p = os.path.join(folder, "feedback_item_final.csv")
    wanted = ["Fund Tickers", "Fund Names", "Fund Amount Allocation", "Comment"]
    try:
        if os.path.exists(p):
            fb = pd.read_csv(p)
            drop_cols = [
                c for c in ["username", "reviewed_at", "row_index"] if c in fb.columns
            ]
            fb_view = fb.drop(columns=drop_cols, errors="ignore")
            if not fb_view.empty:
                # Ensure all wanted columns exist
                for c in wanted:
                    if c not in fb_view.columns:
                        fb_view[c] = "" if c != "Fund Amount Allocation" else 0.0
                return fb_view[wanted]
    except Exception:
        pass
    df = item_final_df.copy()
    if df is None or df.empty:
        df = pd.DataFrame(
            {
                "Fund Tickers": [""],
                "Fund Names": [""],
                "Fund Amount Allocation": [0.0],
            }
        )
    if "Comment" not in df.columns:
        df["Comment"] = ""
    return df[wanted]


# -----------------------------
# Session & Auth
# -----------------------------
if "saved_pdf_path" not in st.session_state:
    st.session_state.saved_pdf_path = ""
if "ran_pipeline" not in st.session_state:
    st.session_state.ran_pipeline = False
if "last_log_folder" not in st.session_state:
    st.session_state.last_log_folder = None
if "signed_in" not in st.session_state:
    st.session_state.signed_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "active_invoice_id" not in st.session_state:
    st.session_state.active_invoice_id = ""

# -----------------------------
# UI — Sign-in
# -----------------------------
st.title("Data Extraction ML System — Mock")
st.caption(
    "Sign in, auto-identify invoice ID on run, overwrite 6 CSVs if already processed, or reload to update feedback."
)

login = st.container(border=True)
with login:
    st.subheader("Sign In")
    if not st.session_state.signed_in:
        u = st.text_input(
            "Username", placeholder="e.g., jdoe", value=st.session_state.username
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            sign_in = st.button("Sign In", type="primary", use_container_width=True)
        with c2:
            clear_name = st.button("Clear", use_container_width=True)
        if clear_name:
            st.session_state.username = ""
            st.rerun()
        if sign_in:
            if not u.strip():
                st.error("Please enter a username.")
            else:
                st.session_state.username = u.strip()
                st.session_state.signed_in = True
                st.success(f"Signed in as **{st.session_state.username}**")
                st.rerun()
    else:
        st.success(f"Signed in as **{st.session_state.username}**")
        if st.button("Sign Out", use_container_width=True):
            for k in [
                "inv_reason_df",
                "item_reason_df",
                "inv_final_df",
                "item_final_df",
                "validation",
                "saved_pdf_path",
                "last_log_folder",
                "ran_pipeline",
                "active_invoice_id",
            ]:
                st.session_state.pop(k, None)
            st.session_state.signed_in = False
            st.session_state.username = ""
            st.rerun()

# -----------------------------
# UI — Reload processed invoice
# -----------------------------
reload_box = st.container(border=True)
with reload_box:
    st.subheader("Reload processed invoice (review & update feedback)")
    ids = list_existing_invoice_ids()
    if not ids:
        st.info(
            "No processed invoices found in logs/. Once you process one, it will appear here."
        )
    else:
        choice = st.selectbox("Select invoice ID", options=ids, index=0)
        if st.button("Load selected invoice", type="primary"):
            folder = invoice_folder(choice)
            try:
                inv_r = pd.read_csv(os.path.join(folder, "invoice_reasoning.csv"))
                item_r = pd.read_csv(os.path.join(folder, "item_reasoning.csv"))
                inv_f = pd.read_csv(os.path.join(folder, "invoice_final.csv"))
                item_f = pd.read_csv(os.path.join(folder, "item_final.csv"))
                ok, alloc_sum, total_amt = validate_totals(inv_f, item_r)
                st.session_state.inv_reason_df = inv_r
                st.session_state.item_reason_df = item_r
                st.session_state.inv_final_df = inv_f
                st.session_state.item_final_df = item_f
                st.session_state.validation = (ok, alloc_sum, total_amt)
                st.session_state.ran_pipeline = True
                st.session_state.last_log_folder = folder
                st.session_state.active_invoice_id = choice
                st.success(f"Loaded invoice **{choice}**.")
            except Exception as e:
                st.error(f"Failed to load invoice {choice}: {e}")

# -----------------------------
# UI — Upload & Process (auto-identify ID; overwrite if exists)
# -----------------------------
controls = st.container(border=True)
with controls:
    st.subheader("Upload & Process (auto-identify invoice ID)")
    c_left, c_right = st.columns([7, 3])

    with c_left:
        pdf_file = st.file_uploader(
            "Drop an invoice PDF",
            type=["pdf"],
            accept_multiple_files=False,
            disabled=not st.session_state.signed_in,
        )

    with c_right:
        process_click = st.button(
            "Process",
            type="primary",
            use_container_width=True,
            disabled=(not st.session_state.signed_in) or (pdf_file is None),
        )

    reset_clicked = st.button(
        "Clear Active Invoice (UI only)",
        use_container_width=True,
        disabled=not st.session_state.signed_in,
    )
    if reset_clicked:
        for k in [
            "inv_reason_df",
            "item_reason_df",
            "inv_final_df",
            "item_final_df",
            "validation",
            "saved_pdf_path",
            "last_log_folder",
            "ran_pipeline",
            "active_invoice_id",
        ]:
            st.session_state.pop(k, None)
        st.rerun()

# Process button
if process_click:
    saved_path = save_uploaded_file(pdf_file)
    st.session_state.saved_pdf_path = saved_path

    # Identify invoice ID and run pipeline
    t0 = time.time()
    seed = _seed_from_pdf(saved_path)
    inv_r_df, item_r_df, invoice_id = mock_adi_extract_from_seed(seed)
    inv_final = to_final_invoice_table(inv_r_df)
    item_final = to_final_items_table(item_r_df)
    ok, alloc_sum, total_amt = validate_totals(inv_final, item_r_df)
    t1 = time.time()
    run_time_sec = round(t1 - t0, 3)
    run_cost_usd = round(0.001 * max(1, len(item_final)), 4)  # mock cost

    folder = invoice_folder(invoice_id)
    already_processed = os.path.isdir(folder) and all(
        os.path.exists(os.path.join(folder, f)) for f in BASE_FILES
    )

    # Overwrite all 6 CSVs by policy
    log_overwrite_base_csvs(folder, inv_r_df, item_r_df, inv_final, item_final)
    init_or_reset_feedback_csvs(folder, st.session_state.username)

    # Append to master log
    append_master_log(
        username=st.session_state.username,
        invoice_id=invoice_id,
        pdf_path=saved_path,
        invoice_final=inv_final,
        validation=(ok, alloc_sum, total_amt),
        run_time_sec=run_time_sec,
        run_cost_usd=run_cost_usd,
    )

    # Load into session
    st.session_state.inv_reason_df = inv_r_df
    st.session_state.item_reason_df = item_r_df
    st.session_state.inv_final_df = inv_final
    st.session_state.item_final_df = item_final
    st.session_state.validation = (ok, alloc_sum, total_amt)
    st.session_state.ran_pipeline = True
    st.session_state.last_log_folder = folder
    st.session_state.active_invoice_id = invoice_id

    if already_processed:
        st.warning(
            f"Invoice **{invoice_id}** was already processed. All 6 CSVs were overwritten."
        )
    else:
        st.success(f"Processed NEW invoice **{invoice_id}**.")

# -----------------------------
# Status
# -----------------------------
status_box = st.container(border=True)
with status_box:
    st.subheader("Pipeline Status")
    if st.session_state.signed_in:
        st.caption(f"User: **{st.session_state.username}**")
    else:
        st.warning("Please sign in with a username to enable actions.")
    if st.session_state.active_invoice_id:
        st.success(f"Active invoice: **{st.session_state.active_invoice_id}**")
    if st.session_state.get("ran_pipeline", False):
        ok, alloc_sum, total_amt = st.session_state.validation
        if ok:
            st.success(
                f"Validation passed: Sum of Fund Amount Allocation ({alloc_sum:,.2f}) matches Total Amount ({total_amt:,.2f})."
            )
        else:
            st.warning(
                f"Validation failed: Sum of Fund Amount Allocation is {alloc_sum:,.2f} but Total Amount is {total_amt:,.2f}."
            )

# -----------------------------
# Reasoning sections (Arrow-safe for display only)
# -----------------------------
if st.session_state.get("ran_pipeline", False):
    with st.expander("AI Reasoning – Invoice-Level", expanded=True):
        with st.container(border=True):
            st.dataframe(
                _arrow_safe(st.session_state.inv_reason_df),
                use_container_width=True,
                height=260,
            )

    with st.expander("AI Reasoning – Invoice-Item", expanded=False):
        with st.container(border=True):
            st.dataframe(
                _arrow_safe(st.session_state.item_reason_df),
                use_container_width=True,
                height=260,
            )

st.markdown("---")

# -----------------------------
# Finalized Outputs
# -----------------------------
st.subheader("Finalized Outputs")
if st.session_state.get("ran_pipeline", False):
    st.markdown("**Invoice-Level (Attributes → Values, single row)**")
    st.container(border=True).dataframe(
        st.session_state.inv_final_df, use_container_width=True
    )

    st.markdown("**Item-Level (Multiple rows, one per item)**")
    st.container(border=True).dataframe(
        st.session_state.item_final_df, use_container_width=True
    )
else:
    st.info("Reload or process an invoice to produce finalized tables.")

# -----------------------------
# Human Feedback (SAME schema + Comment + Username)
# -----------------------------
if st.session_state.get("ran_pipeline", False) and st.session_state.active_invoice_id:
    st.header("Human Feedback")
    folder = invoice_folder(st.session_state.active_invoice_id)

    # Invoice feedback (seed from existing feedback if present)
    with st.expander("Feedback: Invoice Final (single row)", expanded=True):
        inv_fb_view = load_feedback_invoice_or_default(
            folder, st.session_state.inv_final_df
        )
        edited_inv = st.data_editor(
            inv_fb_view,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Total Amount": st.column_config.NumberColumn(
                    "Total Amount", step=0.01, format="%.2f"
                ),
                "Comment": st.column_config.TextColumn("Comment"),
            },
        )
        try:
            total = float(edited_inv.loc[0, "Total Amount"])
            st.caption(f"Current Total Amount (edited): **{total:,.2f}**")
        except Exception:
            pass
        if st.button("Save Invoice Feedback", type="primary"):
            save_feedback_invoice(folder, edited_inv, st.session_state.username)
            st.success(f"Saved: {os.path.join(folder, 'feedback_invoice_final.csv')}")

    # Item feedback (seed from existing feedback if present)
    with st.expander(
        "Feedback: Item Final (multi-row; add/remove allowed)", expanded=False
    ):
        items_fb_view = load_feedback_items_or_default(
            folder, st.session_state.item_final_df
        )
        edited_items = st.data_editor(
            items_fb_view,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Fund Tickers": st.column_config.TextColumn("Fund Tickers"),
                "Fund Names": st.column_config.TextColumn("Fund Names"),
                "Fund Amount Allocation": st.column_config.NumberColumn(
                    "Fund Amount Allocation", step=0.01, format="%.2f"
                ),
                "Comment": st.column_config.TextColumn("Comment"),
            },
        )
        # Live validation vs (possibly edited) invoice total
        try:
            total_amount = float(edited_inv.loc[0, "Total Amount"])
        except Exception:
            try:
                total_amount = float(
                    st.session_state.inv_final_df.loc[0, "Total Amount"]
                )
            except Exception:
                total_amount = None
        items_sum = (
            pd.to_numeric(edited_items["Fund Amount Allocation"], errors="coerce")
            .fillna(0)
            .sum()
        )
        if (total_amount is not None) and round(items_sum, 2) == round(total_amount, 2):
            st.success(
                f"Allocation sum {items_sum:,.2f} matches Total Amount {total_amount:,.2f}."
            )
        else:
            st.warning(
                f"Allocation sum {items_sum:,.2f} does not match Total Amount {total_amount if total_amount is not None else 'N/A'}."
            )

        if st.button(
            "Save Item Feedback", type="primary", disabled=len(edited_items) < 1
        ):
            save_feedback_items(folder, edited_items, st.session_state.username)
            st.success(f"Saved: {os.path.join(folder, 'feedback_item_final.csv')}")

st.markdown("---")

with st.expander("Notes"):
    st.write(
        """
        - Auto-ID invoice on Process (mocked): invoice ID is the extracted invoice number.
        - Canonical folder per invoice: `logs/{invoice_id}/` with exactly **6 CSVs**:
          invoice_reasoning, item_reasoning, invoice_final, item_final, feedback_invoice_final, feedback_item_final.
        - If an invoice already exists, **all 6 CSVs are overwritten** on Process.
        - Use **Reload processed invoice** to review existing artifacts and update feedback (editors seed from feedback if present).
        - `_master_log.csv` is append-only and includes:
          `timestamp, username, invoice_id, pdf_file, invoice_folder, invoice_number, total_amount, items_sum, validation_pass, run_time_sec, run_cost_usd`.
        """
    )
