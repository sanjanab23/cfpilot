"""
ppt_brain.py  ── v3.0
Template-driven PPT pipeline. LLM only writes insights — never decides structure.

ARCHITECTURE:
  - TEMPLATE mode: known object keywords → fixed slide plan + pre-written SOQL
  - CUSTOM mode: anything else → LLM planner (legacy path)

TEMPLATES:
  T1  Leads breakdown/analysis
  T2  Opportunities breakdown/pipeline
  T3  Accounts breakdown
  T4  Invoices & billing analysis

SHARED FIXES CARRIED FORWARD:
  A1  Parallel SOQL execution (ThreadPoolExecutor, 6 workers)
  A2  SOQL result cache (5-min TTL)
  A4  Python-side anomaly detection
  A6  Insight prompt — number + direction + implication
  A8  Slide-level retry (2 attempts)
  DI  Data integrity guard — values validated against raw Salesforce data
  P2  Hard 90-second timeout
  P5  Skipped-slide audit log
  P6  Minimal fallback deck
  NO_FUNNEL  Pyramid/funnel removed
  TL  Quarterly count bar chart (real SF data)
"""

import os
import json
import re
import logging
import ast
import time
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

from salesforce_utils import execute_soql_query
from query_validator import validate_soql_query
from ppt_utils import cache_get, cache_set, cache_clear_expired, detect_anomalies

try:
    from security_utils import LLMInputSanitizer
except ImportError:
    class LLMInputSanitizer:
        @staticmethod
        def sanitize(text): return text

load_dotenv()
logger = logging.getLogger(__name__)

GENERATION_TIMEOUT = 90

# ─── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="xiaomi/mimo-v2-flash",
    temperature=0.3,
    max_retries=3,
    model_kwargs={"extra_body": {"provider": {"zdr": True}}}
)

# ─── STATE ────────────────────────────────────────────────────────────────────
class PPTState(TypedDict):
    user_query:     str
    dashboard_data: str
    raw_plan:       str
    slides:         List[Dict]
    error:          str
    skipped_slides: List[Dict]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _try_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r"\[.*\]", r"\{.*\}"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

def _run_soql(query: str) -> list:
    cached = cache_get(query)
    if cached is not None:
        return cached
    try:
        is_valid, cleaned, msg = validate_soql_query(query)
        if not is_valid:
            logger.warning(f"SOQL invalid: {msg} | {query[:80]}")
            return []
        result_str = execute_soql_query(cleaned)
        if not isinstance(result_str, str):
            return []
        if (result_str.startswith("Error")
                or result_str.startswith("Salesforce Error")
                or result_str.startswith("Execution Error")
                or "does not support aggregate operator" in result_str
                or "ERROR at Row" in result_str
                or result_str == "No records found matching that query."):
            logger.warning(f"SOQL error: {result_str[:100]}")
            return []
        parsed = _try_json(result_str)
        if isinstance(parsed, list):
            cache_set(query, parsed)
            return parsed
        evaled = ast.literal_eval(result_str)
        if isinstance(evaled, list):
            cache_set(query, evaled)
            return evaled
    except Exception as e:
        logger.error(f"SOQL exec error: {e} | {query[:80]}")
    return []

def _first_val(rows: list, default=0):
    """Extract first numeric value from a single-row aggregate result."""
    if not rows:
        return default
    for v in rows[0].values():
        try:
            return int(float(str(v)))
        except (TypeError, ValueError):
            pass
    return default

def _get_nested(data: dict, path: str, default=None):
    """
    Retrieve value from nested dict using dot notation (e.g. 'Account.Name').
    Handles standard nested JSON, flattened keys from some SOQL adapters,
    and Salesforce's behavior of returning only the field name for grouped relationship fields.
    """
    if not data or not path:
        return default
    
    # 1. Try exact match (some adapters flatten keys or use aliasing internally)
    if path in data:
        return data[path]
    
    # 2. Try the last part of the path (Salesforce often returns just the field name for grouped fields)
    parts = path.split(".")
    if len(parts) > 1 and parts[-1] in data:
        return data[parts[-1]]
        
    # 3. Traverse the nested structure
    val = data
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return default
    return val if val is not None else default

def _rows_to_chart(rows: list, label_key: str, value_key: str) -> list:
    """Convert raw SOQL rows to [{label, value}] for chart rendering."""
    result = []
    for row in rows:
        lbl = str(_get_nested(row, label_key, ""))
        val = _get_nested(row, value_key, 0)
        if val is None:
            val = 0
        try:
            result.append({"label": lbl, "value": float(str(val))})
        except (TypeError, ValueError):
            pass
    return result

# ─── MONTHLY TIMELINE HELPER ─────────────────────────────────────────────────

def _monthly_counts(obj: str, date_field: str, obj_label: str,
                    where_extra: str = "") -> Dict:
    """Run 12 SOQL queries to get per-month record counts for the full FY (Apr→Mar)."""
    now  = datetime.now()
    yr, mo = now.year, now.month
    fys  = yr if mo >= 4 else yr - 1
    fye  = fys + 1
    lbl  = f"FY{str(fys)[2:]}-{str(fye)[2:]}"

    # Build list of (label, start_date, end_date) for each month Apr → Mar
    import calendar
    months = [
        (4,fys),(5,fys),(6,fys),
        (7,fys),(8,fys),(9,fys),
        (10,fys),(11,fys),(12,fys),
        (1,fye),(2,fye),(3,fye),
    ]
    month_abbr = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    chart_data = []
    for m, y in months:
        last_day = calendar.monthrange(y, m)[1]
        start = f"{y}-{m:02d}-01T00:00:00Z"
        end   = f"{y}-{m:02d}-{last_day:02d}T23:59:59Z"
        label = f"{month_abbr[m]}-{str(y)[2:]}"
        where = f"WHERE {date_field} >= {start} AND {date_field} <= {end}"
        if where_extra:
            where += f" AND {where_extra}"
        rows = _run_soql(f"SELECT COUNT(Id) cnt FROM {obj} {where}")
        cnt  = _first_val(rows)
        chart_data.append({"label": label, "value": cnt})
        logger.info(f"Monthly '{label}' ({obj_label}): {cnt}")
    return {
        "slide_type":   "chart",
        "chart_type":   "bar",
        "title":        f"{obj_label} Count by Month — {lbl}",
        "x_axis":       "Month",
        "y_axis":       f"Number of {obj_label}",
        "raw_data":     chart_data,
        "data_source":  "Salesforce",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  T1 — LEADS TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_leads_plan(user_query: str) -> List[Dict]:
    """
    T1 — Fixed slide plan for any lead-related query.
    All SOQL queries are pre-written and validated.
    """
    logger.info("T1: Building Leads plan")
    slides = []

    slides.append({
        "slide_type": "cover",
        "title":      "Leads Breakdown & Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI: Total / Dropped / In-Progress / Qualified ─────────────────────
    total_rows    = _run_soql("SELECT COUNT(Id) cnt FROM Lead WHERE CreatedDate = LAST_N_DAYS:365")
    dropped_rows  = _run_soql("SELECT COUNT(Id) cnt FROM Lead WHERE Status = 'Drop' AND CreatedDate = LAST_N_DAYS:365")
    inprog_rows   = _run_soql("SELECT COUNT(Id) cnt FROM Lead WHERE IsConverted = false AND Status != 'Drop' AND CreatedDate = LAST_N_DAYS:365")
    qual_rows     = _run_soql("SELECT COUNT(Id) cnt FROM Lead WHERE IsConverted = true AND CreatedDate = LAST_N_DAYS:365")
    kpi_data = [
        {"label": "Total Leads (12M)",      "value": _first_val(total_rows)},
        {"label": "Dropped Leads",           "value": _first_val(dropped_rows)},
        {"label": "In-Progress Leads",       "value": _first_val(inprog_rows)},
        {"label": "Qualified / Converted",   "value": _first_val(qual_rows)},
    ]
    slides.append({"slide_type": "kpi", "title": "Lead Summary — Last 12 Months", "raw_data": kpi_data})

    # ── Leads by Status/Stage ───────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Status, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY Status ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Leads by Status / Stage", "x_axis": "Status", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Status", "cnt"),
    })

    # ── Leads by Country ────────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Country, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND Country != null "
        "GROUP BY Country ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Leads by Country", "x_axis": "Country", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Country", "cnt"),
    })

    # ── Leads by State ──────────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT State, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND State != null "
        "GROUP BY State ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Leads by State", "x_axis": "State", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "State", "cnt"),
    })

    # ── Salesperson Analysis (table with per-owner counts) ──────────────────
    rows = _run_soql(
        "SELECT Owner.Name, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY Owner.Name ORDER BY COUNT(Id) DESC LIMIT 20"
    )
    owner_table = []
    for r in rows:
        owner = _get_nested(r, "Owner.Name", "")
        total = int(r.get("cnt", 0) or 0)
        owner_esc = owner.replace("'", "\\'")
        # Get converted count for this owner
        conv_rows = _run_soql(
            f"SELECT COUNT(Id) cnt FROM Lead "
            f"WHERE Owner.Name = '{owner_esc}' AND IsConverted = true "
            f"AND CreatedDate = LAST_N_DAYS:365"
        )
        conv = _first_val(conv_rows)
        # Get dropped count for this owner
        drop_rows = _run_soql(
            f"SELECT COUNT(Id) cnt FROM Lead "
            f"WHERE Owner.Name = '{owner_esc}' AND Status = 'Drop' "
            f"AND CreatedDate = LAST_N_DAYS:365"
        )
        drop = _first_val(drop_rows)
        conv_pct = f"{round(conv / total * 100, 1)}%" if total else "0%"
        owner_table.append({
            "Salesperson":  owner,
            "Total Leads":  total,
            "Converted":    conv,
            "Dropped":      drop,
            "Conv. Rate":   conv_pct,
        })
    slides.append({"slide_type": "table", "title": "Salesperson / Owner Analysis", "raw_data": owner_table})

    # ── Lead Source Analysis ────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT LeadSource, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND LeadSource != null "
        "GROUP BY LeadSource ORDER BY COUNT(Id) DESC LIMIT 12"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Lead Source Analysis",
        "raw_data": _rows_to_chart(rows, "LeadSource", "cnt"),
    })

    # ── Dropped Reason Analysis ─────────────────────────────────────────────
    # Attempt to group by the custom reason field.
    # Note: Description is Long Text and cannot be grouped in SOQL.
    rows = _run_soql(
        "SELECT TCL_Drop_Reason__c, COUNT(Id) cnt FROM Lead "
        "WHERE Status = 'Drop' AND CreatedDate = LAST_N_DAYS:365 "
        "AND TCL_Drop_Reason__c != null "
        "GROUP BY TCL_Drop_Reason__c ORDER BY COUNT(Id) DESC LIMIT 12"
    )
    if rows:
        slides.append({
            "slide_type": "chart", "chart_type": "bar",
            "title": "Dropped Lead — Reason Analysis", "x_axis": "Reason", "y_axis": "Count",
            "raw_data": _rows_to_chart(rows, "TCL_Drop_Reason__c", "cnt"),
        })
    else:
        logger.warning("Skipping 'Dropped Lead — Reason Analysis': TCL_Drop_Reason__c missing or no data.")

    # ── Leads by Industry ───────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Industry, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND Industry != null "
        "GROUP BY Industry ORDER BY COUNT(Id) DESC LIMIT 12"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Leads by Industry", "x_axis": "Industry", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Industry", "cnt"),
    })

    # ── Leads by Rating ─────────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Rating, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND Rating != null "
        "GROUP BY Rating ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Leads by Rating",
        "raw_data": _rows_to_chart(rows, "Rating", "cnt"),
    })

    # ── Lead Velocity — monthly count (FY) ──────────────────────────────────
    slides.append(_monthly_counts("Lead", "CreatedDate", "Leads"))

    # ── Detailed Conversion Table ────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Status, COUNT(Id) cnt FROM Lead "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY Status ORDER BY COUNT(Id) DESC"
    )
    total_all = _first_val(total_rows)
    conv_table = []
    for r in rows:
        status = r.get("Status", "")
        cnt    = int(r.get("cnt", 0) or 0)
        pct    = f"{round(cnt / total_all * 100, 1)}%" if total_all else "0%"
        conv_table.append({"Status": status, "Count": cnt, "% of Total": pct})
    slides.append({"slide_type": "table", "title": "Detailed Lead Conversion Table", "raw_data": conv_table})

    # ── So What ─────────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T2 — OPPORTUNITIES TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_opportunities_plan(user_query: str) -> List[Dict]:
    """T2 — Fixed slide plan for opportunity / pipeline queries."""
    logger.info("T2: Building Opportunities plan")
    slides = []

    slides.append({
        "slide_type": "cover",
        "title":      "Opportunities Breakdown & Pipeline Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI Cards ───────────────────────────────────────────────────────────
    total_rows  = _run_soql("SELECT COUNT(Id) cnt FROM Opportunity WHERE CreatedDate = LAST_N_DAYS:365")
    amt_rows    = _run_soql("SELECT SUM(Amount) amt FROM Opportunity WHERE CreatedDate = LAST_N_DAYS:365")
    qty_rows    = _run_soql("SELECT SUM(TotalOpportunityQuantity) qty FROM Opportunity WHERE CreatedDate = LAST_N_DAYS:365")
    won_rows    = _run_soql("SELECT SUM(Amount) amt FROM Opportunity WHERE StageName = 'Closed Won' AND CreatedDate = LAST_N_DAYS:365")
    kpi_data = [
        {"label": "Total Opportunities",    "value": _first_val(total_rows)},
        {"label": "Total Pipeline Amount",  "value": _first_val(amt_rows)},
        {"label": "Total Quantity",         "value": _first_val(qty_rows)},
        {"label": "Closed Won Amount",      "value": _first_val(won_rows)},
    ]
    slides.append({"slide_type": "kpi", "title": "Opportunity Summary — Last 12 Months", "raw_data": kpi_data})

    # ── Pipeline by Stage (count) ────────────────────────────────────────────
    rows = _run_soql(
        "SELECT StageName, COUNT(Id) cnt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY StageName ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Pipeline by Stage", "x_axis": "Stage", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "StageName", "cnt"),
    })

    # ── Monthly Volume/Amount Trend (line) ───────────────────────────────────
    rows = _run_soql(
        "SELECT CALENDAR_YEAR(CreatedDate) yr, CALENDAR_MONTH(CreatedDate) mo, "
        "SUM(Amount) amt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY CALENDAR_YEAR(CreatedDate), CALENDAR_MONTH(CreatedDate) "
        "ORDER BY CALENDAR_YEAR(CreatedDate), CALENDAR_MONTH(CreatedDate)"
    )
    monthly = [
        {"label": f"{r.get('yr','')}-{str(r.get('mo','')).zfill(2)}",
         "value": float(r.get("amt", 0) or 0)}
        for r in rows
    ]
    slides.append({
        "slide_type": "chart", "chart_type": "line",
        "title": "Monthly Pipeline Amount Trend", "x_axis": "Month", "y_axis": "Amount",
        "raw_data": monthly,
    })

    # ── Opps by Type ─────────────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Type, COUNT(Id) cnt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND Type != null "
        "GROUP BY Type ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Opportunities by Type",
        "raw_data": _rows_to_chart(rows, "Type", "cnt"),
    })

    # ── Opps by Reporting Region ─────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Account.TCL_Reporting_Region__c, COUNT(Id) cnt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "AND Account.TCL_Reporting_Region__c != null "
        "GROUP BY Account.TCL_Reporting_Region__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Opportunities by Reporting Region", "x_axis": "Region", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Account.TCL_Reporting_Region__c", "cnt"),
    })

    # ── Opps by Record Type ──────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT RecordType.Name, COUNT(Id) cnt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY RecordType.Name ORDER BY COUNT(Id) DESC LIMIT 10"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Opportunities by Record Type", "x_axis": "Record Type", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "RecordType.Name", "cnt"),
    })

    # ── Opps by Sales Group ──────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT TCL_Sales_Group__c, COUNT(Id) cnt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND TCL_Sales_Group__c != null "
        "GROUP BY TCL_Sales_Group__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Opportunities by Sales Group", "x_axis": "Sales Group", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "TCL_Sales_Group__c", "cnt"),
    })

    # ── Top Accounts by Pipeline (table) ─────────────────────────────────────
    rows = _run_soql(
        "SELECT Account.Name, COUNT(Id) cnt, SUM(Amount) amt FROM Opportunity "
        "WHERE CreatedDate = LAST_N_DAYS:365 AND Account.Name != null "
        "GROUP BY Account.Name ORDER BY SUM(Amount) DESC LIMIT 15"
    )
    acct_table = [
        {"Account": _get_nested(r, "Account.Name", ""), "Opportunities": r.get("cnt",0), "Pipeline Amount": r.get("amt",0) or 0}
        for r in rows
    ]
    slides.append({"slide_type": "table", "title": "Top Accounts by Pipeline Value", "raw_data": acct_table})

    # ── Age in Days distribution (bucketed) ──────────────────────────────────
    rows = _run_soql(
        "SELECT AgeInDays, COUNT(Id) cnt FROM Opportunity "
        "WHERE IsClosed = false AND CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY AgeInDays ORDER BY AgeInDays ASC LIMIT 200"
    )
    buckets = {"0-30 days": 0, "31-60 days": 0, "61-90 days": 0,
               "91-180 days": 0, "180+ days": 0}
    for r in rows:
        age = int(r.get("AgeInDays", 0) or 0)
        cnt = int(r.get("cnt", 0) or 0)
        if age <= 30:     buckets["0-30 days"] += cnt
        elif age <= 60:   buckets["31-60 days"] += cnt
        elif age <= 90:   buckets["61-90 days"] += cnt
        elif age <= 180:  buckets["91-180 days"] += cnt
        else:             buckets["180+ days"] += cnt
    age_data = [{"label": k, "value": v} for k, v in buckets.items()]
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Open Opportunities by Age (Days)", "x_axis": "Age Bucket", "y_axis": "Count",
        "raw_data": age_data,
    })

    # ── Monthly Opportunity count (FY) ───────────────────────────────────────
    slides.append(_monthly_counts("Opportunity", "CreatedDate", "Opportunities"))

    # ── So What ──────────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T3 — ACCOUNTS TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_accounts_plan(user_query: str) -> List[Dict]:
    """T3 — Fixed slide plan for account breakdown queries."""
    logger.info("T3: Building Accounts plan")
    slides = []

    slides.append({
        "slide_type": "cover",
        "title":      "Accounts Breakdown & Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI Cards ───────────────────────────────────────────────────────────
    total_rows  = _run_soql("SELECT COUNT(Id) cnt FROM Account")
    new_q_rows  = _run_soql("SELECT COUNT(Id) cnt FROM Account WHERE CreatedDate = THIS_QUARTER")
    new_yr_rows = _run_soql("SELECT COUNT(Id) cnt FROM Account WHERE CreatedDate = LAST_N_DAYS:365")
    reg_rows    = _run_soql("SELECT COUNT(Id) cnt FROM Account WHERE TCL_Reporting_Region__c != null")
    kpi_data = [
        {"label": "Total Accounts",          "value": _first_val(total_rows)},
        {"label": "New This Quarter",         "value": _first_val(new_q_rows)},
        {"label": "New Last 12 Months",       "value": _first_val(new_yr_rows)},
        {"label": "Accounts with Region",     "value": _first_val(reg_rows)},
    ]
    slides.append({"slide_type": "kpi", "title": "Account Summary", "raw_data": kpi_data})

    # ── Accounts by BP Grouping ──────────────────────────────────────────────
    rows = _run_soql(
        "SELECT TCL_BP_Grouping__c, COUNT(Id) cnt FROM Account "
        "WHERE TCL_BP_Grouping__c != null "
        "GROUP BY TCL_BP_Grouping__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Accounts by BP Grouping", "x_axis": "BP Grouping", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "TCL_BP_Grouping__c", "cnt"),
    })

    # ── Accounts by Reporting Region ─────────────────────────────────────────
    rows = _run_soql(
        "SELECT TCL_Reporting_Region__c, COUNT(Id) cnt FROM Account "
        "WHERE TCL_Reporting_Region__c != null "
        "GROUP BY TCL_Reporting_Region__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Accounts by Reporting Region",
        "raw_data": _rows_to_chart(rows, "TCL_Reporting_Region__c", "cnt"),
    })

    # ── Accounts by Division ─────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT TCL_Division_Desc__c, COUNT(Id) cnt FROM Account "
        "WHERE TCL_Division_Desc__c != null "
        "GROUP BY TCL_Division_Desc__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Accounts by Division", "x_axis": "Division", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "TCL_Division_Desc__c", "cnt"),
    })

    # ── Accounts by Type ─────────────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Type, COUNT(Id) cnt FROM Account "
        "WHERE Type != null "
        "GROUP BY Type ORDER BY COUNT(Id) DESC LIMIT 10"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Accounts by Type",
        "raw_data": _rows_to_chart(rows, "Type", "cnt"),
    })

    # ── Accounts by Owner (table) ─────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Owner.Name, TCL_Reporting_Region__c, COUNT(Id) cnt FROM Account "
        "WHERE Owner.Name != null "
        "GROUP BY Owner.Name, TCL_Reporting_Region__c "
        "ORDER BY COUNT(Id) DESC LIMIT 20"
    )
    owner_table = [
        {"Salesperson": _get_nested(r, "Owner.Name", ""), "Region": _get_nested(r, "TCL_Reporting_Region__c", ""), "Accounts": r.get("cnt", 0)}
        for r in rows
    ]
    slides.append({"slide_type": "table", "title": "Accounts by Salesperson / Owner (Top 20)", "raw_data": owner_table})

    # ── Monthly new accounts (FY) ────────────────────────────────────────────
    slides.append(_monthly_counts("Account", "CreatedDate", "Accounts"))

    # ── So What ──────────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T4 — INVOICES & BILLING TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_invoices_plan(user_query: str) -> List[Dict]:
    """T4 — Fixed slide plan for invoice / billing analysis."""
    logger.info("T4: Building Invoices plan")
    slides = []

    now     = datetime.now()
    yr, mo  = now.year, now.month
    fys     = yr if mo >= 4 else yr - 1
    fye     = fys + 1
    fy_s    = f"{fys}-04-01"
    fy_e    = f"{fye}-03-31"
    fy_lbl  = f"FY{str(fys)[2:]}-{str(fye)[2:]}"

    slides.append({
        "slide_type": "cover",
        "title":      "Invoices & Billing Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI Cards ───────────────────────────────────────────────────────────
    inv_fy_rows   = _run_soql(f"SELECT COUNT(Id) cnt FROM TCL_Invoice__c WHERE TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice_Date__c <= {fy_e}")
    vol_fy_rows   = _run_soql(f"SELECT SUM(TCL_Billing_Quantity_Metric_TON__c) qty FROM TCL_Invoice_Line_Item__c WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e}")
    inv_30_rows   = _run_soql("SELECT COUNT(Id) cnt FROM TCL_Invoice__c WHERE TCL_Invoice_Date__c = LAST_N_DAYS:30")
    inv_12m_rows  = _run_soql("SELECT COUNT(Id) cnt FROM TCL_Invoice__c WHERE TCL_Invoice_Date__c = LAST_N_DAYS:365")
    kpi_data = [
        {"label": f"Total Invoices ({fy_lbl})",    "value": _first_val(inv_fy_rows)},
        {"label": "Total Billing Volume (MT)",      "value": _first_val(vol_fy_rows)},
        {"label": "Invoices Last 30 Days",          "value": _first_val(inv_30_rows)},
        {"label": "Invoices Last 12 Months",        "value": _first_val(inv_12m_rows)},
    ]
    slides.append({"slide_type": "kpi", "title": f"Billing Summary — {fy_lbl}", "raw_data": kpi_data})

    # ── Actual vs Target: Division-wise (bullet chart) ───────────────────────
    actual_div = _run_soql(
        f"SELECT TCL_Invoice__r.TCL_Division_Description__c, "
        f"SUM(TCL_Billing_Quantity_Metric_TON__c) qty "
        f"FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} "
        f"AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Invoice__r.TCL_Division_Description__c != null "
        f"GROUP BY TCL_Invoice__r.TCL_Division_Description__c "
        f"ORDER BY SUM(TCL_Billing_Quantity_Metric_TON__c) DESC LIMIT 15"
    )
    target_div = _run_soql(
        f"SELECT TCL_Division_Description__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Division_Description__c != null "
        f"GROUP BY TCL_Division_Description__c LIMIT 15"
    )
    actual_dmap = {_get_nested(r, "TCL_Invoice__r.TCL_Division_Description__c", ""): float(r.get("qty", 0) or 0) for r in actual_div}
    target_dmap = {_get_nested(r, "TCL_Division_Description__c", ""):               float(r.get("tgt", 0) or 0) for r in target_div}
    all_divs    = sorted(set(list(actual_dmap) + list(target_dmap)) - {""})
    avt_div     = []
    for d in all_divs:
        avt_div.append({"label": f"{d} - Actual", "value": actual_dmap.get(d, 0)})
        avt_div.append({"label": f"{d} - Target", "value": target_dmap.get(d, 0)})
    slides.append({
        "slide_type": "chart", "chart_type": "bullet",
        "title": f"Actual vs Target — Division-wise (Billing Volume MT) — {fy_lbl}",
        "x_axis": "Division", "y_axis": "Volume (MT)",
        "raw_data": avt_div,
    })

    # ── Actual vs Target: Region-wise (bullet chart) ─────────────────────────
    actual_reg = _run_soql(
        f"SELECT TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c, "
        f"SUM(TCL_Billing_Quantity_Metric_TON__c) qty "
        f"FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} "
        f"AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c != null "
        f"GROUP BY TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c"
    )
    target_reg = _run_soql(
        f"SELECT TCL_Region__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Region__c != null "
        f"GROUP BY TCL_Region__c"
    )
    actual_rmap = {_get_nested(r, "TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c", ""): float(r.get("qty", 0) or 0) for r in actual_reg}
    target_rmap = {_get_nested(r, "TCL_Region__c", ""):                float(r.get("tgt", 0) or 0) for r in target_reg}
    all_regs    = sorted(set(list(actual_rmap) + list(target_rmap)) - {""})
    avt_reg     = []
    for reg in all_regs:
        avt_reg.append({"label": f"{reg} - Actual", "value": actual_rmap.get(reg, 0)})
        avt_reg.append({"label": f"{reg} - Target", "value": target_rmap.get(reg, 0)})
    slides.append({
        "slide_type": "chart", "chart_type": "bullet",
        "title": f"Actual vs Target — Region-wise (Billing Volume MT) — {fy_lbl}",
        "x_axis": "Region", "y_axis": "Volume (MT)",
        "raw_data": avt_reg,
    })

    # ── Invoices by Region (pie) ──────────────────────────────────────────────
    rows = _run_soql(
        f"SELECT TCL_Account__r.TCL_Reporting_Region__c, COUNT(Id) cnt "
        f"FROM TCL_Invoice__c "
        f"WHERE TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Account__r.TCL_Reporting_Region__c != null "
        f"GROUP BY TCL_Account__r.TCL_Reporting_Region__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Invoices by Reporting Region",
        "raw_data": _rows_to_chart(rows, "TCL_Account__r.TCL_Reporting_Region__c", "cnt"),
    })

    # ── Invoices by Division (bar) ────────────────────────────────────────────
    rows = _run_soql(
        f"SELECT TCL_Division_Description__c, COUNT(Id) cnt "
        f"FROM TCL_Invoice__c "
        f"WHERE TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Division_Description__c != null "
        f"GROUP BY TCL_Division_Description__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Invoices by Division", "x_axis": "Division", "y_axis": "Invoice Count",
        "raw_data": _rows_to_chart(rows, "TCL_Division_Description__c", "cnt"),
    })

    # ── Monthly Invoice Count Trend (line) ────────────────────────────────────
    rows = _run_soql(
        f"SELECT CALENDAR_YEAR(TCL_Invoice_Date__c) yr, CALENDAR_MONTH(TCL_Invoice_Date__c) mo, COUNT(Id) cnt "
        f"FROM TCL_Invoice__c "
        f"WHERE TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice_Date__c <= {fy_e} "
        f"GROUP BY CALENDAR_YEAR(TCL_Invoice_Date__c), CALENDAR_MONTH(TCL_Invoice_Date__c) "
        f"ORDER BY CALENDAR_YEAR(TCL_Invoice_Date__c), CALENDAR_MONTH(TCL_Invoice_Date__c)"
    )
    monthly = [
        {"label": f"{r.get('yr','')}-{str(r.get('mo','')).zfill(2)}", "value": float(r.get("cnt",0) or 0)}
        for r in rows
    ]
    slides.append({
        "slide_type": "chart", "chart_type": "line",
        "title": "Monthly Invoice Count Trend", "x_axis": "Month", "y_axis": "Invoice Count",
        "raw_data": monthly,
    })

    # ── Top Accounts by Billing Volume (table) ────────────────────────────────
    rows = _run_soql(
        f"SELECT TCL_Invoice__r.TCL_Account__r.Name, "
        f"SUM(TCL_Billing_Quantity_Metric_TON__c) qty, "
        f"COUNT(TCL_Invoice__c) inv_cnt "
        f"FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} "
        f"AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Invoice__r.TCL_Account__r.Name != null "
        f"GROUP BY TCL_Invoice__r.TCL_Account__r.Name "
        f"ORDER BY SUM(TCL_Billing_Quantity_Metric_TON__c) DESC LIMIT 15"
    )
    acct_table = [
        {"Account": _get_nested(r, "TCL_Invoice__r.TCL_Account__r.Name", ""),
         "Billing Vol (MT)": r.get("qty", 0) or 0, "Invoice Count": r.get("inv_cnt", 0) or 0}
        for r in rows
    ]
    slides.append({"slide_type": "table", "title": "Top Accounts by Billing Volume (MT)", "raw_data": acct_table})

    # ── Billing Volume by Division (bar) ──────────────────────────────────────
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Billing Volume (MT) by Division", "x_axis": "Division", "y_axis": "Volume (MT)",
        "raw_data": _rows_to_chart(actual_div, "TCL_Invoice__r.TCL_Division_Description__c", "qty"),
    })

    # ── Monthly Invoice count (FY) ───────────────────────────────────────────
    slides.append(_monthly_counts("TCL_Invoice__c", "TCL_Invoice_Date__c", "Invoices"))

    # ── So What ──────────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T5 — QUOTES TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_quotes_plan(user_query: str) -> List[Dict]:
    """T5 — Fixed slide plan for quote / quotation / proposal queries."""
    logger.info("T5: Building Quotes plan")
    slides = []

    slides.append({
        "slide_type": "cover",
        "title":      "Quotes & Proposals Breakdown Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI Cards ───────────────────────────────────────────────────────────
    total_rows  = _run_soql("SELECT COUNT(Id) cnt FROM Quote WHERE CreatedDate = LAST_N_DAYS:365")
    val_rows    = _run_soql("SELECT SUM(TotalPrice) amt FROM Quote WHERE CreatedDate = LAST_N_DAYS:365")
    sent_rows   = _run_soql("SELECT COUNT(Id) cnt FROM Quote WHERE Status = 'Sent' AND CreatedDate = LAST_N_DAYS:365")
    acc_rows    = _run_soql("SELECT COUNT(Id) cnt FROM Quote WHERE Status = 'Accepted' AND CreatedDate = LAST_N_DAYS:365")
    
    kpi_data = [
        {"label": "Total Quotes (12M)",     "value": _first_val(total_rows)},
        {"label": "Total Quote Value",      "value": _first_val(val_rows)},
        {"label": "Status: Sent",           "value": _first_val(sent_rows)},
        {"label": "Status: Accepted",       "value": _first_val(acc_rows)},
    ]
    slides.append({"slide_type": "kpi", "title": "Quote Summary — Last 12 Months", "raw_data": kpi_data})

    # ── Quotes by Status (pie) ──────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Status, COUNT(Id) cnt FROM Quote "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY Status ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Quotes by Status",
        "raw_data": _rows_to_chart(rows, "Status", "cnt"),
    })

    # ── Quotes by Region (bar) ──────────────────────────────────────────────
    # Region is accessed via Opportunity -> Account
    rows = _run_soql(
        "SELECT Opportunity.Account.TCL_Reporting_Region__c, COUNT(Id) cnt FROM Quote "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "AND Opportunity.Account.TCL_Reporting_Region__c != null "
        "GROUP BY Opportunity.Account.TCL_Reporting_Region__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Quotes by Reporting Region", "x_axis": "Region", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Opportunity.Account.TCL_Reporting_Region__c", "cnt"),
    })

    # ── Quotes by Owner (table) ─────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Owner.Name, COUNT(Id) cnt, SUM(TotalPrice) amt FROM Quote "
        "WHERE CreatedDate = LAST_N_DAYS:365 "
        "GROUP BY Owner.Name ORDER BY SUM(TotalPrice) DESC LIMIT 15"
    )
    owner_table = [
        {"Salesperson": _get_nested(r, "Owner.Name", ""), "Quotes": r.get("cnt", 0), "Total Value": r.get("amt", 0) or 0}
        for r in rows
    ]
    slides.append({"slide_type": "table", "title": "Top Salespeople by Quote Value", "raw_data": owner_table})

    # ── Monthly Quote volume (FY) ───────────────────────────────────────────
    slides.append(_monthly_counts("Quote", "CreatedDate", "Quotes"))

    # ── So What ──────────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T6 — TARGET VS ACTUAL TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_targets_plan(user_query: str) -> List[Dict]:
    """T6 — Fixed slide plan for target vs actual / ABP tracking / achievement queries."""
    logger.info("T6: Building Target vs Actual plan")
    slides = []

    now     = datetime.now()
    yr, mo  = now.year, now.month
    fys     = yr if mo >= 4 else yr - 1
    fye     = fys + 1
    fy_s    = f"{fys}-04-01"
    fy_e    = f"{fye}-03-31"
    fy_lbl  = f"FY{str(fys)[2:]}-{str(fye)[2:]}"

    slides.append({
        "slide_type": "cover",
        "title":      "Target vs Actual — Performance Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI: Total Target, Total Actual, Achievement %, Gap ──────────────
    tgt_total_rows = _run_soql(
        f"SELECT SUM(TCL_Target__c) tgt FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e}"
    )
    act_total_rows = _run_soql(
        f"SELECT SUM(TCL_Billing_Quantity_Metric_TON__c) qty FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e}"
    )
    total_target = _first_val(tgt_total_rows)
    total_actual = _first_val(act_total_rows)
    achievement  = round(total_actual / total_target * 100, 1) if total_target else 0
    gap          = total_target - total_actual

    kpi_data = [
        {"label": f"Total Target ({fy_lbl})",   "value": total_target},
        {"label": "Total Actual (Billing MT)",   "value": total_actual},
        {"label": "Achievement %",               "value": f"{achievement}%"},
        {"label": "Gap (Target - Actual)",       "value": gap},
    ]
    slides.append({"slide_type": "kpi", "title": f"Target vs Actual Summary — {fy_lbl}", "raw_data": kpi_data})

    # ── Actual vs Target by Division (grouped bar) ───────────────────────
    actual_div = _run_soql(
        f"SELECT TCL_Invoice__r.TCL_Division_Description__c, "
        f"SUM(TCL_Billing_Quantity_Metric_TON__c) qty "
        f"FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} "
        f"AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Invoice__r.TCL_Division_Description__c != null "
        f"GROUP BY TCL_Invoice__r.TCL_Division_Description__c "
        f"ORDER BY SUM(TCL_Billing_Quantity_Metric_TON__c) DESC LIMIT 15"
    )
    target_div = _run_soql(
        f"SELECT TCL_Division_Description__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Division_Description__c != null "
        f"GROUP BY TCL_Division_Description__c LIMIT 15"
    )
    actual_dmap = {_get_nested(r, "TCL_Invoice__r.TCL_Division_Description__c", ""): float(r.get("qty", 0) or 0) for r in actual_div}
    target_dmap = {_get_nested(r, "TCL_Division_Description__c", ""): float(r.get("tgt", 0) or 0) for r in target_div}
    all_divs    = sorted(set(list(actual_dmap) + list(target_dmap)) - {""})
    avt_div     = []
    for d in all_divs:
        avt_div.append({"label": f"{d} - Actual", "value": actual_dmap.get(d, 0)})
        avt_div.append({"label": f"{d} - Target", "value": target_dmap.get(d, 0)})
    slides.append({
        "slide_type": "chart", "chart_type": "bullet",
        "title": f"Actual vs Target — Division-wise (Billing Volume MT) — {fy_lbl}",
        "x_axis": "Division", "y_axis": "Volume (MT)",
        "raw_data": avt_div,
    })

    # ── Actual vs Target by Region (grouped bar) ─────────────────────────
    actual_reg = _run_soql(
        f"SELECT TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c, "
        f"SUM(TCL_Billing_Quantity_Metric_TON__c) qty "
        f"FROM TCL_Invoice_Line_Item__c "
        f"WHERE TCL_Invoice__r.TCL_Invoice_Date__c >= {fy_s} "
        f"AND TCL_Invoice__r.TCL_Invoice_Date__c <= {fy_e} "
        f"AND TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c != null "
        f"GROUP BY TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c"
    )
    target_reg = _run_soql(
        f"SELECT TCL_Region__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Region__c != null "
        f"GROUP BY TCL_Region__c"
    )
    actual_rmap = {_get_nested(r, "TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c", ""): float(r.get("qty", 0) or 0) for r in actual_reg}
    target_rmap = {_get_nested(r, "TCL_Region__c", ""): float(r.get("tgt", 0) or 0) for r in target_reg}
    all_regs    = sorted(set(list(actual_rmap) + list(target_rmap)) - {""})
    avt_reg     = []
    for reg in all_regs:
        avt_reg.append({"label": f"{reg} - Actual", "value": actual_rmap.get(reg, 0)})
        avt_reg.append({"label": f"{reg} - Target", "value": target_rmap.get(reg, 0)})
    slides.append({
        "slide_type": "chart", "chart_type": "bullet",
        "title": f"Actual vs Target — Region-wise (Billing Volume MT) — {fy_lbl}",
        "x_axis": "Region", "y_axis": "Volume (MT)",
        "raw_data": avt_reg,
    })

    # ── Achievement % by Division (bar) ──────────────────────────────────
    ach_div = []
    for d in all_divs:
        act = actual_dmap.get(d, 0)
        tgt = target_dmap.get(d, 0)
        pct = round(act / tgt * 100, 1) if tgt else 0
        ach_div.append({"label": d, "value": pct})
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": f"Achievement % by Division — {fy_lbl}",
        "x_axis": "Division", "y_axis": "Achievement %",
        "raw_data": ach_div,
    })

    # ── Achievement % by Region (bar) ────────────────────────────────────
    ach_reg = []
    for reg in all_regs:
        act = actual_rmap.get(reg, 0)
        tgt = target_rmap.get(reg, 0)
        pct = round(act / tgt * 100, 1) if tgt else 0
        ach_reg.append({"label": reg, "value": pct})
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": f"Achievement % by Region — {fy_lbl}",
        "x_axis": "Region", "y_axis": "Achievement %",
        "raw_data": ach_reg,
    })

    # ── Detailed Division-wise Target vs Actual (table) ──────────────────
    div_table = []
    for d in all_divs:
        act = actual_dmap.get(d, 0)
        tgt = target_dmap.get(d, 0)
        pct = round(act / tgt * 100, 1) if tgt else 0
        div_table.append({
            "Division": d,
            "Target (MT)": round(tgt, 1),
            "Actual (MT)": round(act, 1),
            "Gap (MT)": round(tgt - act, 1),
            "Achievement %": f"{pct}%",
        })
    slides.append({"slide_type": "table", "title": f"Division-wise Target vs Actual — {fy_lbl}", "raw_data": div_table})

    # ── Target vs Actual by Sub-Region (bar) ─────────────────────────────
    target_sub = _run_soql(
        f"SELECT TCL_Sub_Region__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Sub_Region__c != null "
        f"GROUP BY TCL_Sub_Region__c ORDER BY SUM(TCL_Target__c) DESC LIMIT 15"
    )
    sub_data = _rows_to_chart(target_sub, "TCL_Sub_Region__c", "tgt")
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": f"Target by Sub-Region — {fy_lbl}",
        "x_axis": "Sub-Region", "y_axis": "Target (MT)",
        "raw_data": sub_data,
    })

    # ── Target by Month (line trend) ─────────────────────────────────────
    target_monthly = _run_soql(
        f"SELECT TCL_Month_Name__c, SUM(TCL_Target__c) tgt "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Month_Name__c != null "
        f"GROUP BY TCL_Month_Name__c ORDER BY TCL_Month_Name__c"
    )
    monthly_data = _rows_to_chart(target_monthly, "TCL_Month_Name__c", "tgt")
    slides.append({
        "slide_type": "chart", "chart_type": "line",
        "title": f"Target by Month — {fy_lbl}",
        "x_axis": "Month", "y_axis": "Target (MT)",
        "raw_data": monthly_data,
    })

    # ── Volume vs Value vs Forecast (table) ──────────────────────────────
    vol_val_rows = _run_soql(
        f"SELECT TCL_Division_Description__c, SUM(TCL_Target__c) tgt, "
        f"SUM(TCL_Volume__c) vol, SUM(TCL_Value__c) val, SUM(TCL_Forecast__c) fcast "
        f"FROM TCL_ABP_Tracking__c "
        f"WHERE TCL_Transaction_Date__c >= {fy_s} AND TCL_Transaction_Date__c <= {fy_e} "
        f"AND TCL_Division_Description__c != null "
        f"GROUP BY TCL_Division_Description__c ORDER BY SUM(TCL_Target__c) DESC LIMIT 15"
    )
    vvf_table = [
        {
            "Division": _get_nested(r, "TCL_Division_Description__c", ""),
            "Target": r.get("tgt", 0) or 0,
            "Volume": r.get("vol", 0) or 0,
            "Value": r.get("val", 0) or 0,
            "Forecast": r.get("fcast", 0) or 0,
        }
        for r in vol_val_rows
    ]
    slides.append({"slide_type": "table", "title": f"Target / Volume / Value / Forecast — {fy_lbl}", "raw_data": vvf_table})

    # ── So What ──────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  T7 — VISITS & VOC TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_visits_plan(user_query: str) -> List[Dict]:
    """T7 — Fixed slide plan for visit, VOC, adherence, and sentiment queries."""
    logger.info("T7: Building Visits & VOC plan")
    slides = []

    slides.append({
        "slide_type": "cover",
        "title":      "Visit & VOC Analysis",
        "subtitle":   "Tata Chemicals Limited",
    })

    # ── KPI: Total Visits, Planned, Actual, Adherence % ──────────────────
    total_rows   = _run_soql("SELECT COUNT(Id) cnt FROM TCL_Visit__c")
    planned_rows = _run_soql("SELECT COUNT(Id) cnt FROM TCL_Visit__c WHERE Visit_Planned_Date__c != null")
    actual_rows  = _run_soql("SELECT COUNT(Id) cnt FROM TCL_Visit__c WHERE Actual_Visit_Date__c != null")

    total_visits   = _first_val(total_rows)
    planned_visits = _first_val(planned_rows)
    actual_visits  = _first_val(actual_rows)
    adherence_pct  = round(actual_visits / planned_visits * 100, 1) if planned_visits else 0

    kpi_data = [
        {"label": "Total Visits",            "value": total_visits},
        {"label": "Planned Visits",          "value": planned_visits},
        {"label": "Actual Visits",           "value": actual_visits},
        {"label": "Adherence %",             "value": f"{adherence_pct}%"},
    ]
    slides.append({"slide_type": "kpi", "title": "Visit Summary", "raw_data": kpi_data})

    # ── Visit Adherence: Planned vs Actual (bullet chart) ────────────────
    adherence_data = [
        {"label": "Planned Visits", "value": planned_visits},
        {"label": "Actual Visits",  "value": actual_visits},
    ]
    slides.append({
        "slide_type": "chart", "chart_type": "bullet",
        "title": f"Visit Adherence — {adherence_pct}%",
        "x_axis": "Category", "y_axis": "Count",
        "raw_data": adherence_data,
    })

    # ── Visit Status Breakdown (pie) ─────────────────────────────────────
    rows = _run_soql(
        "SELECT Visit_Status__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Visit_Status__c != null "
        "GROUP BY Visit_Status__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Visit Status Breakdown",
        "raw_data": _rows_to_chart(rows, "Visit_Status__c", "cnt"),
    })

    # ── VOC Category Distribution (bar) ──────────────────────────────────
    rows = _run_soql(
        "SELECT VOC_Category__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE VOC_Category__c != null "
        "GROUP BY VOC_Category__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "VOC Category Distribution",
        "x_axis": "VOC Category", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "VOC_Category__c", "cnt"),
    })

    # ── VOC Sub-Category Distribution (bar) ──────────────────────────────
    rows = _run_soql(
        "SELECT VOC_Sub_Category__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE VOC_Sub_Category__c != null "
        "GROUP BY VOC_Sub_Category__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "VOC Sub-Category Distribution",
        "x_axis": "VOC Sub-Category", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "VOC_Sub_Category__c", "cnt"),
    })

    # ── Visits by Type (bar) ─────────────────────────────────────────────
    rows = _run_soql(
        "SELECT Type__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Type__c != null "
        "GROUP BY Type__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Visits by Type",
        "x_axis": "Visit Type", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Type__c", "cnt"),
    })

    # ── Customer Sentiment Distribution (pie) ────────────────────────────
    rows = _run_soql(
        "SELECT Customer_sentiment__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Customer_sentiment__c != null "
        "GROUP BY Customer_sentiment__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "pie",
        "title": "Customer Sentiment Distribution",
        "raw_data": _rows_to_chart(rows, "Customer_sentiment__c", "cnt"),
    })

    # ── Satisfaction Score Distribution (bar) ─────────────────────────────
    rows = _run_soql(
        "SELECT Satisfaction_score__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Satisfaction_score__c != null "
        "GROUP BY Satisfaction_score__c ORDER BY COUNT(Id) DESC"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Satisfaction Score Distribution",
        "x_axis": "Score", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "Satisfaction_score__c", "cnt"),
    })

    # ── Visits by Country/Region (bar) ───────────────────────────────────
    rows = _run_soql(
        "SELECT TCL_Country__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE TCL_Country__c != null "
        "GROUP BY TCL_Country__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    slides.append({
        "slide_type": "chart", "chart_type": "bar",
        "title": "Visits by Country / Region",
        "x_axis": "Country", "y_axis": "Count",
        "raw_data": _rows_to_chart(rows, "TCL_Country__c", "cnt"),
    })

    # ── Adherence by Country (table) ─────────────────────────────────────
    planned_by_country = _run_soql(
        "SELECT TCL_Country__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Visit_Planned_Date__c != null AND TCL_Country__c != null "
        "GROUP BY TCL_Country__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    actual_by_country = _run_soql(
        "SELECT TCL_Country__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Actual_Visit_Date__c != null AND TCL_Country__c != null "
        "GROUP BY TCL_Country__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    planned_map = {_get_nested(r, "TCL_Country__c", ""): int(r.get("cnt", 0) or 0) for r in planned_by_country}
    actual_map  = {_get_nested(r, "TCL_Country__c", ""): int(r.get("cnt", 0) or 0) for r in actual_by_country}
    all_countries = sorted(set(list(planned_map) + list(actual_map)) - {""})
    adherence_table = []
    for c in all_countries:
        p = planned_map.get(c, 0)
        a = actual_map.get(c, 0)
        pct = round(a / p * 100, 1) if p else 0
        adherence_table.append({
            "Country": c,
            "Planned": p,
            "Actual": a,
            "Adherence %": f"{pct}%",
        })
    slides.append({"slide_type": "table", "title": "Visit Adherence by Country", "raw_data": adherence_table})

    # ── Top Customers by Visit Count (table) ─────────────────────────────
    rows = _run_soql(
        "SELECT Customer_Name__c, COUNT(Id) cnt FROM TCL_Visit__c "
        "WHERE Customer_Name__c != null "
        "GROUP BY Customer_Name__c ORDER BY COUNT(Id) DESC LIMIT 15"
    )
    cust_table = [
        {"Customer": _get_nested(r, "Customer_Name__c", ""), "Visit Count": r.get("cnt", 0)}
        for r in rows
    ]
    slides.append({"slide_type": "table", "title": "Top Customers by Visit Count", "raw_data": cust_table})

    # ── Monthly Visit Count (FY) ─────────────────────────────────────────
    slides.append(_monthly_counts("TCL_Visit__c", "Visit_Planned_Date__c", "Visits"))

    # ── So What ──────────────────────────────────────────────────────────
    slides.append({"slide_type": "bullets", "title": "So What? — Recommended Actions",
                   "bullets": [], "raw_data": kpi_data})
    slides.append({"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"})
    return slides


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_TEMPLATE_MAP: List[Tuple[List[str], Any]] = [
    (["lead", "leads"],                                                _build_leads_plan),
    (["opportunity", "opportunities", "pipeline", "opp", "opps"],    _build_opportunities_plan),
    (["account", "accounts"],                                          _build_accounts_plan),
    (["actual vs target", "actual vs. target", "target vs actual",
      "target vs. actual", "target achievement", "abp", "target"],   _build_targets_plan),
    (["invoice", "invoices", "billing", "bill"],                       _build_invoices_plan),
    (["visit", "visits", "voc", "adherence", "sentiment"],            _build_visits_plan),
    (["quote", "quotation", "proposal"],                               _build_quotes_plan),
]

def _detect_template(query: str):
    q = query.lower()
    for keywords, builder in _TEMPLATE_MAP:
        if any(k in q for k in keywords):
            return builder
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1: PLANNER
# ══════════════════════════════════════════════════════════════════════════════

def ppt_planner_node(state: PPTState) -> Dict[str, Any]:
    logger.info("--- PPT PLANNER v3.0 ---")
    try:
        query = LLMInputSanitizer.sanitize(state["user_query"])
    except ValueError as e:
        return {"error": f"Invalid input: {e}", "raw_plan": "[]", "slides": [], "skipped_slides": []}

    cache_clear_expired()

    # ── TEMPLATE MODE ─────────────────────────────────────────────────────────
    template_fn = _detect_template(query)
    if template_fn:
        logger.info(f"TEMPLATE mode: {template_fn.__name__}")
        try:
            plan = template_fn(query)
            return {"raw_plan": json.dumps(plan), "error": "", "skipped_slides": []}
        except Exception as e:
            logger.error(f"Template build failed: {e} — falling back to LLM")

    # ── CUSTOM (LLM) MODE ─────────────────────────────────────────────────────
    logger.info("CUSTOM mode: LLM planner")
    dashboard_ctx = state.get("dashboard_data", "")

    planner_prompt = f"""
You are PPT Planner — create structured data plans for business presentations.

Final slide (before Thank You): slide_type="bullets", title="So What? — Recommended Actions", 4-5 action bullets.

SLIDE TYPES: cover, kpi (use soql_list), chart (bar|line|pie|donut|waterfall|bullet|treemap|scatter|combo), split, table, bullets, thankyou
DO NOT use pyramid or funnel slides.

SOQL RULES (strict):
- No UNION ALL, CASE WHEN, SELECT *, CALENDAR_DAY
- No field aliases on non-aggregate fields
- ORDER BY COUNT(Id) DESC — never ORDER BY alias
- No TCL_Reporting_Region__c on Lead
- No Amount on TCL_Invoice__c
- No GROUP BY on multipicklist or __r relationship paths
- MULTIPICKLIST fields: INCLUDES() / EXCLUDES() only
- LIMIT ≤ 200, every non-aggregate SELECT field must be in GROUP BY

KPI FORMAT: {{"slide_type":"kpi","title":"...","soql_list":[{{"query":"SELECT COUNT(Id) cnt FROM ...","metric_label":"..."}}]}}

Return JSON array ONLY. 8-14 slides.

DASHBOARD CONTEXT: {dashboard_ctx}
USER REQUEST: {query}
"""

    resp = llm.invoke([SystemMessage(content=planner_prompt)])
    plan = _try_json(resp.content.strip())

    if not isinstance(plan, list):
        return {
            "raw_plan": json.dumps([
                {"slide_type": "cover", "title": query[:60]},
                {"slide_type": "bullets", "title": "Notice",
                 "bullets": ["Data processing encountered an error. Please try again."]},
                {"slide_type": "thankyou"},
            ]),
            "error": "Planner fallback", "skipped_slides": [],
        }

    # A1: Parallel SOQL
    jobs = []
    for idx, slide in enumerate(plan):
        sl = slide.pop("soql_list", None)
        sq = slide.pop("soql", None)
        if sl or sq:
            jobs.append((idx, sl, sq))
        else:
            slide["raw_data"] = []

    def _exec_job(job):
        idx, sl, sq = job
        slide = plan[idx]
        st = slide.get("slide_type", "")
        try:
            if sl and isinstance(sl, list):
                if st == "kpi":
                    kpi = []
                    for item in sl:
                        q = item.get("query","") if isinstance(item, dict) else str(item)
                        l = item.get("metric_label", f"Metric {len(kpi)+1}") if isinstance(item, dict) else f"Metric {len(kpi)+1}"
                        rows = _run_soql(q)
                        kpi.append({"label": l, "value": _first_val(rows)})
                    return idx, kpi
                else:
                    merged = []
                    for q in sl:
                        merged.extend(_run_soql(q if isinstance(q, str) else q.get("query","")))
                    return idx, merged
            elif sq:
                return idx, _run_soql(sq)
            return idx, []
        except Exception as e:
            logger.error(f"Job {idx} failed: {e}")
            return idx, []

    with ThreadPoolExecutor(max_workers=6) as ex:
        fmap = {ex.submit(_exec_job, job): job for job in jobs}
        for fut in as_completed(fmap, timeout=GENERATION_TIMEOUT):
            try:
                idx, result = fut.result()
                plan[idx]["raw_data"] = result
            except Exception as e:
                plan[fmap[fut][0]]["raw_data"] = []
                logger.error(f"Parallel job error: {e}")

    # Inject monthly timeline chart if not already present
    if not any("month" in s.get("title","").lower() or "quarter" in s.get("title","").lower() for s in plan):
        tl  = _monthly_counts("Opportunity", "CreatedDate", "Opportunities")
        ins = next((j for j in range(len(plan)-1,-1,-1)
                    if plan[j].get("slide_type") not in ("bullets","thankyou")), len(plan)-1)
        plan = plan[:ins+1] + [tl] + plan[ins+1:]

    return {"raw_plan": json.dumps(plan), "error": "", "skipped_slides": []}


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 2: INSIGHT
# ══════════════════════════════════════════════════════════════════════════════

def ppt_insight_node(state: PPTState) -> Dict[str, Any]:
    logger.info("--- PPT INSIGHT v3.0 ---")

    plan = _try_json(state.get("raw_plan", "[]"))
    if not isinstance(plan, list):
        return {"slides": [], "error": "Invalid plan", "skipped_slides": []}

    polished = []
    skipped  = list(state.get("skipped_slides", []))
    all_kpi  = [item for s in plan if s.get("slide_type") == "kpi"
                for item in s.get("raw_data", [])]
    # Build enriched context for So What: add chart slide summaries
    all_chart_context = []
    for s in plan:
        if s.get("slide_type") == "chart" and s.get("raw_data"):
            for d in s["raw_data"][:3]:  # top 3 data points per chart
                all_chart_context.append({
                    "label": f"{s.get('title','')} — {d.get('label','')}",
                    "value": d.get("value", 0)
                })
    combined_kpi_context = all_kpi + all_chart_context

    for slide in plan:
        st       = slide.get("slide_type", "bullets")
        title    = slide.get("title", "")
        raw_data = slide.get("raw_data", [])

        if st in ("cover", "section", "thankyou"):
            polished.append({"slide_type": st, "title": title, "subtitle": slide.get("subtitle","")})
            continue

        if st == "kpi" and raw_data and isinstance(raw_data[0], dict) \
                and "label" in raw_data[0] and "value" in raw_data[0]:
            polished.append({"slide_type": "kpi", "title": title,
                              "data": raw_data,
                              "insights": _generate_kpi_insights(title, raw_data)})
            continue

        if st == "bullets":
            bullets = slide.get("bullets", [])
            if not bullets:
                bullets = _generate_so_what(title, combined_kpi_context, plan)
            polished.append({"slide_type": "bullets", "title": title,
                              "bullets": bullets, "data": [], "insights": [],
                              "data_source": "Salesforce"})
            continue

        if not raw_data:
            skipped.append({"title": title, "slide_type": st, "reason": "No data from Salesforce"})
            logger.warning(f"Skip '{title}': no data")
            continue

        # A4 anomaly + A6/DI insight
        flags    = detect_anomalies(raw_data)
        enriched = None
        for attempt in range(2):
            try:
                # Use dedicated prompt for tables to avoid 2-column simplification
                if st == "table":
                    prompt = _table_insight_prompt(title, raw_data, _format_anomalies(flags))
                else:
                    prompt = _insight_prompt(st, title, slide.get("chart_type","bar"), raw_data, _format_anomalies(flags))
                
                resp     = llm.invoke([SystemMessage(content=prompt)])
                enriched = _try_json(resp.content.strip())
                
                if enriched and isinstance(enriched, dict):
                    if st == "table":
                        # For tables, preserve ALL columns and original names by bypassing chart enforcement
                        enriched["data"] = raw_data
                        enriched["data_source"] = "Salesforce"
                        enriched.setdefault("slide_type", "table")
                        enriched.setdefault("title", title)
                    else:
                        enriched = _enforce_data_integrity(enriched, raw_data)
                    break
                enriched = None
            except Exception as e:
                logger.error(f"Insight error '{title}' attempt {attempt+1}: {e}")
                enriched = None

        if enriched:
            enriched.setdefault("slide_type", st)
            enriched.setdefault("title", title)
            polished.append(enriched)
        else:
            polished.append(_fallback_slide(slide))

    logger.info(f"Insight: {len(polished)} polished, {len(skipped)} skipped")
    return {"slides": polished, "error": "", "skipped_slides": skipped}


# ─── INSIGHT HELPERS ──────────────────────────────────────────────────────────

def _generate_kpi_insights(title: str, kpi_data: list) -> List[str]:
    try:
        resp   = llm.invoke([SystemMessage(content=f"""
Senior analyst at Tata Chemicals. KPI values from Salesforce:
{json.dumps(kpi_data, indent=2)}

Write 2-3 insight chips for "{title}".
Format: "[NUMBER] [DIRECTION] — [IMPLICATION]". Max 15 words each.
Use ONLY numbers from the data above. 

**Privacy & DPA Compliance:**
- ALLOWED: Names of individuals.
- FORBIDDEN: Any mention of phone numbers or email addresses.
- If data contains contact info, generalize or omit it.

Return JSON: {{"insights": ["...","...","..."]}}
""")])
        parsed = _try_json(resp.content.strip())
        if parsed and isinstance(parsed.get("insights"), list):
            return parsed["insights"][:3]
    except Exception as e:
        logger.error(f"KPI insight error: {e}")
    return []


def _generate_so_what(title: str, kpi_context: list, full_plan: list) -> List[str]:
    try:
        resp   = llm.invoke([SystemMessage(content=f"""
Senior analyst writing "{title}" slide for Tata Chemicals CEO presentation.
Key metrics from Salesforce data:
{json.dumps(kpi_context, indent=2) if kpi_context else "No KPI data"}

Write 4-5 concrete action bullets based ONLY on the numbers above. Format: "Action verb + specific action + outcome".
Reference specific numbers where possible. Max 20 words each.

**Privacy & DPA Compliance:**
- ALLOWED: Names of individuals.
- FORBIDDEN: Any mention of phone numbers or email addresses.
- Ensure zero leakage of contact details in the output text.

Return JSON: {{"bullets": ["...","...","...","...","..."]}}
""")])
        parsed = _try_json(resp.content.strip())
        if parsed and isinstance(parsed.get("bullets"), list):
            return parsed["bullets"]
    except Exception as e:
        logger.error(f"So What error: {e}")
    return [
        "Review top-performing segments and double down on acquisition in those areas.",
        "Address data gaps in null/unknown categories to improve reporting accuracy.",
        "Schedule quarterly business reviews aligned to the trends in this analysis.",
        "Set up automated alerts for any metric that deviates more than 20% from trend.",
    ]


def _format_anomalies(flags: dict) -> str:
    if not flags:
        return "No significant anomalies."
    parts = []
    for a in flags.get("anomalies", []):
        parts.append(f"ANOMALY: {a['label']} z={a['z_score']} val={a['value']:.0f}")
    for dm in flags.get("mom_deltas", [])[:3]:
        parts.append(f"MoM: {dm['from']}→{dm['to']}: {dm['delta_pct']:+.1f}%")
    if flags.get("pareto_pct"):
        parts.append(f"PARETO: top {flags['pareto_pct']:.0f}% items = 80% of total")
    if flags.get("max_label"):
        parts.append(f"PEAK: {flags['max_label']}={flags['max_value']:.0f}")
        parts.append(f"TROUGH: {flags['min_label']}={flags['min_value']:.0f}")
    return "\n".join(parts) if parts else "No significant anomalies."

def _table_insight_prompt(title, raw_data, anomaly_ctx) -> str:
    return f"""You are PPT Insight Analyst briefing the CEO of Tata Chemicals.

SLIDE: Type=table | Title={title}

RAW SALESFORCE DATA (ONLY source you may reference):
{json.dumps(raw_data[:50], indent=2)}

PRE-COMPUTED STATS:
{anomaly_ctx}

⚠ CRITICAL: Look at the RAW DATA and write textual insights. 
Do NOT transform the table structure - the system will handle the table display.
Your job is to provide context and "So What?" bullets based on the actual names and numbers in the table.

**Privacy & DPA Compliance:**
- ALLOWED: Names of individuals.
- FORBIDDEN: Any mention of phone numbers or email addresses.
- If the table contains contact info, strictly avoid mentioning those specific values in your insights.

INSIGHT FORMAT: "[NUMBER FROM DATA] [DIRECTION] — [BUSINESS IMPLICATION]"
- Max 18 words per bullet
- 3 bullets (2 if fewer than 4 data points)

Return JSON only:
{{
  "slide_type": "table",
  "title": "{title}",
  "bullets": [],
  "insights": ["...","...","..."]
}}
"""


def _insight_prompt(st, title, chart_type, raw_data, anomaly_ctx) -> str:
    # Detect object type for prompt grounding
    q = title.lower()
    obj = "opportunities" if "opp" in q or "pipeline" in q else ("invoices" if "inv" in q or "bill" in q else "leads")
    
    return f"""You are PPT Insight Analyst briefing the CEO of Tata Chemicals.

SLIDE: Type={st} | Title={title} | Chart={chart_type} | Category={obj}

RAW SALESFORCE DATA (ONLY source you may reference):
{json.dumps(raw_data[:50], indent=2)}

PRE-COMPUTED STATS:
{anomaly_ctx}

⚠ CRITICAL: Use ONLY numbers from RAW SALESFORCE DATA. Never invent values.
Every insight must cite a specific number from the data.
Refer to these records as {obj} (never swap terminology as "leads").

**Privacy & DPA Compliance:**
- ALLOWED: Names of individuals.
- FORBIDDEN: Any mention of phone numbers or email addresses.
- If the data contains contact info, omit it from your textual insights.

INSIGHT FORMAT: "[NUMBER FROM DATA] [DIRECTION] — [BUSINESS IMPLICATION]"
- Max 18 words per bullet
- 3 bullets (2 if fewer than 4 data points)

Return JSON only:
{{
  "slide_type": "{st}",
  "title": "{title}",
  "chart_type": "{chart_type}",
  "x_axis": "...",
  "y_axis": "...",
  "data": [{{"label": "...", "value": 0}}],
  "bullets": [],
  "insights": ["...","...","..."]
}}

"data" must contain ONLY items from RAW SALESFORCE DATA with unchanged values."""


def _enforce_data_integrity(enriched: Dict, raw_data: list) -> Dict:
    # Always stamp data source
    enriched["data_source"] = "Salesforce"
    if not raw_data or not isinstance(raw_data[0], dict):
        return enriched
    if not enriched.get("data"):
        enriched["data"] = _map_raw_to_chart(raw_data)
        return enriched
    raw_map: Dict[str, float] = {}
    for row in raw_data:
        keys = list(row.keys())
        lk = next((k for k in keys if k.lower() in
                   ("label","status","stagename","name","region","owner","leadsource",
                    "source","type","division","category","month","year","product")),
                  keys[0] if keys else None)
        vk = next((k for k in keys if k.lower() in
                   ("value","count","cnt","amount","total","sum","volume","qty",
                    "quantity","target","actual")),
                  keys[-1] if len(keys) > 1 else (keys[0] if keys else None))
        if lk and vk and lk != vk:
            try:
                # Use _get_nested here to extract values from nested SOQL fields
                lv = str(_get_nested(row, lk, "")).lower()
                vv = float(str(_get_nested(row, vk, 0)) or "0")
                if lv:
                    raw_map[lv] = vv
            except (TypeError, ValueError):
                pass
    corrected = []
    for item in enriched.get("data", []):
        lbl = str(item.get("label",""))
        val = item.get("value", 0)
        rv  = raw_map.get(lbl.lower())
        if rv is not None and abs(float(val) - rv) > 0.01:
            logger.warning(f"DI: '{lbl}' corrected {val} → {rv}")
            item = dict(item); item["value"] = rv
        corrected.append(item)
    enriched["data"] = corrected
    return enriched


def _map_raw_to_chart(raw: list) -> list:
    if not raw or not isinstance(raw[0], dict):
        return []
    result = []
    for row in raw[:30]:
        keys = list(row.keys())
        lk = next((k for k in keys if k.lower() in
                   ("label","status","stagename","name","region","owner","leadsource",
                    "source","type","division","category","month","year","product")),
                  keys[0] if keys else None)
        vk = next((k for k in keys if k.lower() in
                   ("value","count","cnt","amount","total","sum","volume","qty",
                    "quantity","target","actual")),
                  keys[-1] if len(keys) > 1 else (keys[0] if keys else None))
        if not lk or not vk or lk == vk:
            continue
        
        # Use _get_nested here to extract values from nested SOQL fields
        lv = str(_get_nested(row, lk, ""))
        rv = _get_nested(row, vk, 0)
        
        if rv is None:
            rv = 0
        try:
            result.append({"label": lv, "value": float(str(rv))})
        except (TypeError, ValueError):
            pass
    return result


def _fallback_slide(slide: Dict) -> Dict:
    return {
        "slide_type": slide.get("slide_type","bullets"),
        "title": slide.get("title",""),
        "chart_type": slide.get("chart_type","bar"),
        "x_axis": slide.get("x_axis",""),
        "y_axis": slide.get("y_axis",""),
        "data": _map_raw_to_chart(slide.get("raw_data",[])),
        "bullets": slide.get("bullets",[]),
        "insights": [],
    }


# ─── GRAPH ────────────────────────────────────────────────────────────────────

workflow = StateGraph(PPTState)
workflow.add_node("ppt_planner", ppt_planner_node)
workflow.add_node("ppt_insight", ppt_insight_node)
workflow.set_entry_point("ppt_planner")
workflow.add_edge("ppt_planner", "ppt_insight")
workflow.add_edge("ppt_insight", END)
ppt_brain = workflow.compile()

# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def generate_slide_plan(user_query: str, dashboard_data: str = "") -> List[Dict]:
    inputs = {"user_query": user_query, "dashboard_data": dashboard_data,
              "raw_plan": "", "slides": [], "error": "", "skipped_slides": []}

    def _run():
        return ppt_brain.invoke(inputs)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            result = fut.result(timeout=GENERATION_TIMEOUT)
        except FuturesTimeout:
            logger.error(f"Timed out after {GENERATION_TIMEOUT}s")
            return _fallback_deck(user_query)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return _fallback_deck(user_query)

    skipped = result.get("skipped_slides", [])
    if skipped:
        logger.info(f"Skipped: {[s['title'] for s in skipped]}")
    return result.get("slides") or _fallback_deck(user_query)


def _fallback_deck(query: str) -> List[Dict]:
    return [
        {"slide_type": "cover", "title": query[:60], "subtitle": "Tata Chemicals Limited"},
        {"slide_type": "bullets", "title": "System Notice",
         "bullets": ["Presentation could not be generated within the time limit.",
                     "Please try again or simplify your request."],
         "data": [], "insights": []},
        {"slide_type": "thankyou", "subtitle": "Tata Chemicals Limited"},
    ]