"""
test.py — Salesforce Data Inspector
=====================================
Run this to check WHAT data is actually coming back from Salesforce:
  - Which fields are populated vs null
  - The raw structure (nested objects, relationship fields)
  - Any field name surprises

Usage:
    python test.py
    python test.py --object Lead --limit 2
    python test.py --soql "SELECT Id, Name FROM Account LIMIT 3"
"""

import json
import sys
import argparse
from collections import defaultdict

try:
    from salesforce_utils import execute_soql_query
except ImportError:
    print("❌ Could not import salesforce_utils. Make sure you're running from the project root.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PRESET QUERIES — edit these to probe different objects
# ─────────────────────────────────────────────────────────────────────────────
PRESET_QUERIES = {

    # ── Leads ────────────────────────────────────────────────────────────────
    "lead_sample": """
        SELECT Id, Name, Company, Status, LeadSource, Rating,
               Owner.Name, CreatedDate, IsConverted, TCL_Dropped__c,
               TCL_Product_Business__c, TCL_Customer_Segment__c,
               TCL_Days_In_New__c, TCL_Days_In_Qualified__c,
               TCL_Days_to_Convert__c, TCL_Remarks_for_Drop__c
        FROM Lead
        ORDER BY CreatedDate DESC LIMIT 3
    """,

    # ── Opportunities ────────────────────────────────────────────────────────
    "opp_sample": """
        SELECT Id, Name, StageName, Amount, CloseDate, Probability,
               Account.Name, Account.TCL_Reporting_Region__c,
               Owner.Name, AgeInDays, LastStageChangeDate,
               TCL_Divison__c, TCL_Region__c, TCL_Loss_Reason__c,
               TCL_Number_of_Quotes__c, TCL_Opportunity_Cycle_Time__c
        FROM Opportunity
        ORDER BY CreatedDate DESC LIMIT 3
    """,

    # ── Visits ───────────────────────────────────────────────────────────────
    "visit_sample": """
        SELECT Id, Name, Visit_Status__c, Type__c,
               Customer_Name__r.Name,
               Actual_Visit_Date__c, Visit_Planned_Date__c,
               VOC_Category__c, VOC_Sub_Category__c, VOC__c,
               Customer_sentiment__c, Satisfaction_score__c,
               Key_highlights__c, Visit_MOM__c,
               Customer_top_5_priorities__c,
               Internal_Discussion_Remarks__c,
               Objective_of_Visit__c, Place_of_Visit__c,
               Owner.Name
        FROM TCL_Visit__c
        ORDER BY Actual_Visit_Date__c DESC LIMIT 3
    """,

    # ── Accounts ─────────────────────────────────────────────────────────────
    "account_sample": """
        SELECT Id, Name, TCL_SAP_Code__c, TCL_BP_Grouping__c,
               TCL_Reporting_Region__c, TCL_Division_Desc__c,
               Owner.Name, TCL_Customer_Status__c,
               TCL_Sales_Group__c, TCL_Region__c,
               TCL_IS_Active__c, Industry
        FROM Account
        LIMIT 3
    """,

    # ── Orders ───────────────────────────────────────────────────────────────
    "order_sample": """
        SELECT Id, OrderNumber, Status, EffectiveDate,
               Account.Name, Account.Owner.Name,
               TCL_Total_Amount__c, TCL_Quantity__c,
               Total_Line_Metric_Ton__c,
               TCL_Order_Type_Description__c,
               TCL_Delivery_Status__c,
               TCL_Requested_Delivery_Date__c,
               TCL_Overall_SD_Process_Status__c,
               TCL_PO_Number__c, CurrencyIsoCode
        FROM Order
        ORDER BY EffectiveDate DESC LIMIT 3
    """,

    # ── Invoices ─────────────────────────────────────────────────────────────
    "invoice_sample": """
        SELECT Id, Name, TCL_Invoice_Number__c, TCL_Invoice_Date__c,
               TCL_Account__r.Name, TCL_Account__r.Owner.Name,
               TCL_Invoice_Amount__c, TCL_Net_Amount__c,
               TCL_Division_Description__c,
               TCL_Overall_Processing_Status__c,
               TCL_Billing_Document_IsCancelled__c,
               Total_Line_Metric_Ton__c,
               TCL_Payment_Terms__c, Month__c, Year__c,
               CurrencyIsoCode
        FROM TCL_Invoice__c
        ORDER BY TCL_Invoice_Date__c DESC LIMIT 3
    """,

    # ── Invoice Line Items ───────────────────────────────────────────────────
    "invoice_line_sample": """
        SELECT Id, Name,
               TCL_Billing_Quantity_Metric_TON__c,
               TCL_Invoice__r.TCL_Invoice_Number__c,
               TCL_Invoice__r.TCL_Invoice_Date__c,
               TCL_Invoice__r.TCL_Division_Description__c,
               TCL_Invoice__r.TCL_Account__r.Name,
               TCL_Invoice__r.TCL_Account__r.TCL_Reporting_Region__c,
               TCL_Material_Description__c,
               TCL_Net_Amount__c, TCL_Gross_Amount__c,
               Unit_of_Measure__c
        FROM TCL_Invoice_Line_Item__c
        ORDER BY TCL_Invoice__r.TCL_Invoice_Date__c DESC LIMIT 3
    """,

    # ── Quotes ───────────────────────────────────────────────────────────────
    "quote_sample": """
        SELECT Id, QuoteNumber, Status,
               Opportunity.Name, Opportunity.Account.Name,
               Opportunity.Account.TCL_Reporting_Region__c,
               Owner.Name, TotalPrice, GrandTotal,
               TCL_SAP_Quotation_Number__c, TCL_Quantity__c,
               TCL_QuoteAccepted__c, TCL_QuoteApproved__c,
               TCL_Send_to_SAP__c, ExpirationDate, CreatedDate
        FROM Quote
        ORDER BY CreatedDate DESC LIMIT 3
    """,

    # ── Cases ────────────────────────────────────────────────────────────────
    "case_sample": """
        SELECT Id, CaseNumber, Status, Origin, Type,
               Account.Name, Account.Owner.Name,
               TCL_Product_category__c, TCL_Product_Sub_Category__c,
               TCL_Complaint_category__c, TCL_Complaint_Sub_category__c,
               TCL_Affected_Quantity__c, CreatedDate, ClosedDate
        FROM Case
        ORDER BY CreatedDate DESC LIMIT 3
    """,

    # ── Customer Onboarding ──────────────────────────────────────────────────
    "onboarding_sample": """
        SELECT Id, Name, Account__r.Name, Account__r.Owner.Name,
               Account__r.TCL_Reporting_Region__c,
               Account__r.TCL_Division_Desc__c,
               Status__c, TCL_Onboarding_Type__c,
               TCL_BP_Grouping__c, CreatedDate
        FROM TCL_CustomerOnBoarding__c
        ORDER BY CreatedDate DESC LIMIT 3
    """,

    # ── Target Tracking ──────────────────────────────────────────────────────
    "target_sample": """
        SELECT Id, TCL_Target__c, TCL_Volume__c, TCL_Value__c,
               TCL_Division_Description__c, TCL_Region__c,
               TCL_Sub_Region__c, TCL_Month__c, TCL_Year__c,
               TCL_Transaction_Date__c, TCL_Fiscal_Year__c
        FROM TCL_ABP_Tracking__c
        ORDER BY TCL_Transaction_Date__c DESC LIMIT 3
    """,

    # ── Visits — MOM specifically ────────────────────────────────────────────
    "visit_mom_sample": """
        SELECT Id, Name, Customer_Name__r.Name,
               Actual_Visit_Date__c, Visit_Status__c,
               Visit_MOM__c, VOC_Category__c, Owner.Name
        FROM TCL_Visit__c
        WHERE Visit_MOM__c != null
        ORDER BY Actual_Visit_Date__c DESC LIMIT 3
    """,

    # ── Support Tracker ──────────────────────────────────────────────────────
    "support_sample": """
        SELECT Id, Status__c, Summary__c, CreatedDate,
               TCL_Business_Owner__r.Name,
               TCL_Business_Owner__r.Account.Name
        FROM TCL_Support_Tracker__c
        ORDER BY CreatedDate DESC LIMIT 3
    """,

}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pretty(v, indent=0) -> str:
    pad = "  " * indent
    if isinstance(v, dict):
        lines = ["{"]
        for k, val in v.items():
            if k == "attributes":
                continue
            lines.append(f"{pad}  {k!r}: {_pretty(val, indent + 1)}")
        lines.append(pad + "}")
        return "\n".join(lines)
    if isinstance(v, list):
        if len(v) == 0:
            return "[]"
        lines = ["["]
        for item in v:
            lines.append(pad + "  " + _pretty(item, indent + 1))
        lines.append(pad + "]")
        return "\n".join(lines)
    return repr(v)


def _flatten_keys(record: dict, prefix="") -> dict:
    """Flatten nested dicts (relationship fields) into dotted key paths."""
    result = {}
    for k, v in record.items():
        if k == "attributes":
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "attributes" in v:
            # It's a relationship object — flatten it
            result.update(_flatten_keys({kk: vv for kk, vv in v.items() if kk != "attributes"}, full_key))
        elif isinstance(v, dict):
            result.update(_flatten_keys(v, full_key))
        else:
            result[full_key] = v
    return result


def _field_coverage(records: list) -> dict:
    """Return per-field: total records, populated count, null count, sample values."""
    coverage = defaultdict(lambda: {"total": 0, "populated": 0, "null": 0, "samples": []})
    for rec in records:
        flat = _flatten_keys(rec)
        for field, value in flat.items():
            entry = coverage[field]
            entry["total"] += 1
            if value is None or value == "" or value == []:
                entry["null"] += 1
            else:
                entry["populated"] += 1
                if len(entry["samples"]) < 2:
                    sample = str(value)[:80] + ("…" if len(str(value)) > 80 else "")
                    if sample not in entry["samples"]:
                        entry["samples"].append(sample)
    return dict(coverage)


def _print_coverage(coverage: dict):
    print(f"\n{'FIELD':<55} {'POP':>5} {'NULL':>5}  SAMPLE VALUES")
    print("─" * 120)
    for field, info in sorted(coverage.items()):
        pop  = info["populated"]
        null = info["null"]
        samples = " | ".join(info["samples"]) if info["samples"] else "—"
        flag = "⚠️ " if pop == 0 else "   "
        print(f"{flag}{field:<53} {pop:>5} {null:>5}  {samples}")


def run_query(soql: str, label: str = "CUSTOM"):
    """Execute a SOQL query and print full inspection output."""
    print(f"\n{'='*80}")
    print(f"🔍  QUERY: {label}")
    print(f"{'='*80}")
    print(f"SOQL:\n{soql.strip()}\n")

    raw = execute_soql_query(soql.strip())

    # ── Parse result ─────────────────────────────────────────────────────────
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"❌ Could not parse result as JSON.\nRaw response:\n{raw[:500]}")
            return
    elif isinstance(raw, (list, dict)):
        parsed = raw
    else:
        print(f"❌ Unexpected return type: {type(raw)}\n{raw}")
        return

    # ── Handle dict wrapper (totalSize / records) ─────────────────────────────
    if isinstance(parsed, dict):
        if "records" in parsed:
            total_size = parsed.get("totalSize", "?")
            records = parsed["records"]
            print(f"✅ Salesforce returned dict with 'records'. totalSize={total_size}, records in this batch={len(records)}")
        elif "error" in str(parsed).lower() or "message" in parsed:
            print(f"❌ Salesforce Error:\n{json.dumps(parsed, indent=2)}")
            return
        else:
            records = [parsed]
            print(f"✅ Single record dict returned.")
    elif isinstance(parsed, list):
        records = parsed
        print(f"✅ Salesforce returned list. Record count = {len(records)}")
    else:
        print(f"❌ Unexpected parsed type: {type(parsed)}")
        return

    if len(records) == 0:
        print("ℹ️  No records returned — object may be empty or query too restrictive.")
        return

    # ── Strip attributes ──────────────────────────────────────────────────────
    clean_records = []
    for r in records:
        if isinstance(r, dict):
            clean_records.append({k: v for k, v in r.items() if k != "attributes"})
    records = clean_records

    # ── Print record-by-record ────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"📄  RECORD DUMP ({len(records)} record(s))")
    print(f"{'─'*80}")
    for i, rec in enumerate(records, 1):
        print(f"\n--- Record {i} ---")
        print(_pretty(rec))

    # ── Field coverage table ──────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"📊  FIELD COVERAGE ANALYSIS ({len(records)} record(s))")
    print(f"{'─'*80}")
    coverage = _field_coverage(records)
    _print_coverage(coverage)

    # ── Summary ───────────────────────────────────────────────────────────────
    always_null = [f for f, info in coverage.items() if info["populated"] == 0]
    always_pop  = [f for f, info in coverage.items() if info["null"] == 0]
    print(f"\n⚠️  Always NULL ({len(always_null)}): {', '.join(always_null) or 'none'}")
    print(f"✅  Always Populated ({len(always_pop)}): {', '.join(always_pop) or 'none'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Salesforce field inspector — check what data is actually coming back."
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESET_QUERIES.keys()),
        nargs="+",
        help="Run one or more preset queries (default: run ALL presets)"
    )
    parser.add_argument(
        "--soql", "-s",
        help="Run a custom SOQL query directly"
    )
    parser.add_argument(
        "--account", "-a",
        help="Filter presets to a specific account name (adds LIKE filter)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available preset query names and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable preset queries:")
        for name, soql in PRESET_QUERIES.items():
            first_line = soql.strip().split('\n')[0].strip()
            print(f"  {name:<30} {first_line[:60]}")
        return

    if args.soql:
        run_query(args.soql, label="CUSTOM SOQL")
        return

    # Which presets to run
    presets_to_run = args.preset if args.preset else list(PRESET_QUERIES.keys())

    for name in presets_to_run:
        if name not in PRESET_QUERIES:
            print(f"⚠️  Unknown preset '{name}' — skipping")
            continue

        soql = PRESET_QUERIES[name]

        # Inject account filter if requested
        if args.account:
            account = args.account.replace("'", "\\'")
            # Determine which filter field to use based on query
            soql_upper = soql.upper()
            if "FROM LEAD" in soql_upper:
                filter_clause = f"Company LIKE '%{account}%'"
            elif "FROM TCL_VISIT__C" in soql_upper:
                filter_clause = f"Customer_Name__r.Name LIKE '%{account}%'"
            elif "FROM TCL_INVOICE__C" in soql_upper:
                filter_clause = f"TCL_Account__r.Name LIKE '%{account}%'"
            elif "FROM TCL_INVOICE_LINE_ITEM__C" in soql_upper:
                filter_clause = f"TCL_Invoice__r.TCL_Account__r.Name LIKE '%{account}%'"
            elif "FROM QUOTE" in soql_upper:
                filter_clause = f"Opportunity.Account.Name LIKE '%{account}%'"
            elif "FROM CASE" in soql_upper:
                filter_clause = f"Account.Name LIKE '%{account}%'"
            elif "FROM TCL_CUSTOMERONBOARDING__C" in soql_upper:
                filter_clause = f"Account__r.Name LIKE '%{account}%'"
            else:
                # Default: Account.Name for Order, Opportunity; Name for Account
                if "FROM ACCOUNT" in soql_upper:
                    filter_clause = f"Name LIKE '%{account}%'"
                else:
                    filter_clause = f"Account.Name LIKE '%{account}%'"

            # Inject WHERE or AND
            if "WHERE" in soql_upper:
                soql = soql.rstrip() + f"\n        AND {filter_clause}"
            else:
                # Insert before ORDER/LIMIT or at end
                import re
                soql = re.sub(
                    r'(ORDER BY|LIMIT)',
                    f"WHERE {filter_clause}\n        \\1",
                    soql,
                    count=1,
                    flags=re.IGNORECASE
                )
                if "WHERE" not in soql.upper():
                    soql = soql.rstrip() + f"\n        WHERE {filter_clause}"

        run_query(soql, label=name)


if __name__ == "__main__":
    main()