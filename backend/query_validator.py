"""
SOQL Query Validator
Prevents SQL injection attacks from LLM-generated queries.

FIXES:
  QV1  Auto-strip invalid SELECT field aliases  (e.g. StageName stage → StageName)
       EXCEPTION: CALENDAR_MONTH/YEAR/DAY aggregate aliases are valid — kept.
  QV2  Auto-strip invalid aggregation arithmetic (e.g. SUM(Amount * 0.3) → SUM(Amount))
  QV3  Fix non-aggregatable Satisfaction_score__c — simply DROP the aggregate,
       never attempt GROUP BY conversion (corrupts WHERE LIKE clauses).
  QV4  REMOVED DAY_ONLY() ORDER BY wrap — illegal on non-aggregate queries
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SOQLValidator:
    """Validates SOQL queries to prevent injection attacks."""

    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE',
        'ALTER', 'CREATE', 'EXEC', 'EXECUTE', 'GRANT',
        'REVOKE', 'MERGE', 'UPSERT'
    ]

    FORBIDDEN_FIELDS = [
        'EMAIL', 'PHONE', 'MOBILE', 'MOBILEPHONE',
        'HOMEPHONE', 'OTHERPHONE', 'FAX', 'TCL_EMAIL__C'
    ]

    ALLOWED_KEYWORDS = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY',
        'LIMIT', 'OFFSET', 'HAVING', 'AND', 'OR', 'NOT',
        'IN', 'LIKE', 'INCLUDES', 'EXCLUDES', 'COUNT',
        'SUM', 'AVG', 'MIN', 'MAX', 'CALENDAR_YEAR',
        'CALENDAR_MONTH', 'CALENDAR_DAY', 'ASC', 'DESC',
        'NULL', 'TRUE', 'FALSE', 'TODAY', 'YESTERDAY',
        'LAST_WEEK', 'THIS_WEEK', 'LAST_MONTH', 'THIS_MONTH',
        'LAST_QUARTER', 'THIS_QUARTER', 'LAST_YEAR', 'THIS_YEAR',
        'NEXT_WEEK', 'NEXT_MONTH', 'NEXT_QUARTER', 'NEXT_YEAR',
        'LAST_N_DAYS', 'LAST_N_MONTHS', 'NEXT_N_DAYS', 'NEXT_N_MONTHS',
        'CALENDAR_QUARTER', 'FISCAL_QUARTER', 'FISCAL_YEAR',
        'CONVERTCURRENCY', 'TYPEOF', 'END', 'WHEN', 'THEN', 'ELSE',
        'DAY_ONLY', 'ISNULL', 'ISBLANK',
    ]

    ALLOWED_OBJECTS = [
        # Standard Objects
        'LEAD',
        'OPPORTUNITY',
        'OPPORTUNITIES',
        'ACCOUNT',
        'CONTACT',
        'USER',
        'ORDER',
        'ORDERLINEITEM',
        'PRODUCT2',
        'TASK',
        'EVENT',
        'CASE',
        'CAMPAIGN',
        'QUOTE',
        'QUOTELINEITEM',
        # Custom Objects
        'TCL_VISIT__C',
        'TCL_INVOICE__C',
        'TCL_INVOICE_LINE_ITEM__C',
        'TCL_ABP_TRACKING__C',
        'TCL_SUPPORT_TRACKER__C',
        'TCL_CUSTOMERONBOARDING__C',
        'TCL_ORDER_LINE_ITEMS__C',
        'ORDERAPI__SALES_ORDER__C',
        'ORDERAPI__SALES_ORDER_LINE__C',
        'SURVEY',
    ]

    @staticmethod
    def validate(query: str) -> Tuple[bool, Optional[str]]:
        if not query or not query.strip():
            return False, "Query is empty"

        # Strip comments (-- style) line-by-line
        q_lines = []
        for line in query.splitlines():
            cleaned_line = line.split('--')[0]
            if cleaned_line.strip():
                q_lines.append(cleaned_line)

        query_sanitized = " ".join(q_lines)
        query_upper = query_sanitized.upper().strip()

        if not query_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"

        for keyword in SOQLValidator.DANGEROUS_KEYWORDS:
            if re.search(rf'\b{keyword}\b', query_upper):
                logger.warning(f"Blocked dangerous keyword: {keyword}")
                return False, f"Dangerous operation detected: {keyword}"

        # Strip string literals before checking for forbidden fields
        # so that WHERE Origin = 'Phone' doesn't falsely match PHONE
        query_no_literals = re.sub(r"'[^']*'", "''", query_upper)
        for field in SOQLValidator.FORBIDDEN_FIELDS:
            if re.search(rf'\b{re.escape(field)}\b', query_no_literals):
                logger.warning(f"Blocked PII field: {field}")
                return False, f"Unauthorized access to personal contact info: {field}"

        if '/*' in query or '*/' in query:
            return False, "Block comments (/* */) not allowed"

        if ';' in query_sanitized:
            return False, "Multiple statements not allowed"

        from_pattern = r'\bFROM\s+([\w_]+(?:__[cC])?(?:__[rR])?)'
        from_matches = re.findall(from_pattern, query_upper)

        if not from_matches:
            return False, "No valid FROM clause found"

        for object_name in from_matches:
            if object_name.endswith('__R'):
                continue
            if object_name not in SOQLValidator.ALLOWED_OBJECTS:
                logger.warning(f"Blocked unauthorized object: {object_name}")
                return False, f"Unauthorized object: {object_name}"

        if len(query) > 5000:
            return False, "Query too long (max 5000 characters)"

        if query.count('(') != query.count(')'):
            return False, "Unbalanced parentheses"

        if re.search(r'\bSELECT\s+\*', query_upper):
            return False, "SELECT * is not allowed, specify fields explicitly"

        single_quotes = query.count("'")
        if single_quotes % 2 != 0:
            return False, "Unbalanced quotes in query"

        return True, None

    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Clean up LLM-generated query formatting.
        Applies all auto-fix rules before validation.
        """
        # Remove markdown code blocks
        query = re.sub(r'```(?:sql|soql)?\s*', '', query)
        query = re.sub(r'```\s*', '', query)

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Remove trailing semicolons
        query = query.rstrip(';').strip()

        # ── AUTO-FIX QV2: Aggregation arithmetic ─────────────────────────
        # Strip invalid arithmetic inside aggregate functions.
        # e.g. SUM(Amount * 0.3) → SUM(Amount)
        query = re.sub(
            r'\b(SUM|AVG|MIN|MAX)\s*\(\s*([\w_.]+)\s*[*/+\-]\s*[\w_.]+\s*\)',
            r'\1(\2)',
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX QV3: Non-aggregatable Satisfaction_score__c ─────────
        # Salesforce does not support AVG/SUM/MIN/MAX on Satisfaction_score__c.
        # Simply DROP the aggregate expression from SELECT entirely.
        # Do NOT attempt GROUP BY conversion — that corrupts WHERE LIKE clauses
        # by injecting "GROUP BY Satisfaction_score__c" inside the LIKE string.
        if re.search(r'\b(AVG|SUM|MIN|MAX)\s*\(\s*Satisfaction_score__c\s*\)', query, flags=re.IGNORECASE):
            logger.warning("QV3: Stripping aggregate from non-aggregatable Satisfaction_score__c")

            # Case 1: aggregate appears mid-SELECT with a leading comma
            # e.g. "COUNT(Id) cnt, AVG(Satisfaction_score__c) avg_sat, MAX(...)"
            query = re.sub(
                r',\s*\b(AVG|SUM|MIN|MAX)\s*\(\s*Satisfaction_score__c\s*\)\s*[\w]*',
                '',
                query,
                flags=re.IGNORECASE,
            )

            # Case 2: aggregate appears first in SELECT with a trailing comma
            # e.g. "SELECT AVG(Satisfaction_score__c) avg_sat, COUNT(Id) cnt"
            query = re.sub(
                r'(SELECT\s+)\b(AVG|SUM|MIN|MAX)\s*\(\s*Satisfaction_score__c\s*\)\s*[\w]*\s*,\s*',
                r'\1',
                query,
                flags=re.IGNORECASE,
            )

            # Case 3: aggregate is the only field in SELECT (edge case)
            # e.g. "SELECT AVG(Satisfaction_score__c) FROM ..."
            query = re.sub(
                r'(SELECT\s+)\b(AVG|SUM|MIN|MAX)\s*\(\s*Satisfaction_score__c\s*\)\s*[\w]*\s*(FROM)',
                r'\1COUNT(Id) cnt \3',
                query,
                flags=re.IGNORECASE,
            )

            # Clean up any double commas or trailing commas before FROM
            query = re.sub(r',\s*,', ',', query)
            query = re.sub(r',\s*\bFROM\b', ' FROM', query, flags=re.IGNORECASE)
            query = re.sub(r'SELECT\s*,', 'SELECT ', query, flags=re.IGNORECASE)

        # ── AUTO-FIX QV1: Invalid non-aggregate SELECT aliases ────────────
        # SOQL does not allow aliases on plain SELECT fields.
        # e.g.  StageName stage → StageName
        # EXCEPTION: CALENDAR_MONTH(x) month, CALENDAR_YEAR(x) year,
        #            COUNT(Id) cnt — these are aggregate aliases and are VALID.
        def _strip_plain_alias(match_obj):
            field = match_obj.group(1).strip()
            alias = match_obj.group(2).strip()
            # Keep aliases on aggregate/calendar functions — they contain '('
            if '(' in field:
                return match_obj.group(0)
            # Keep known valid SOQL aggregate alias keywords
            if alias.lower() in ('cnt', 'count', 'value', 'label',
                                  'month', 'year', 'day', 'quarter',
                                  'total', 'amt', 'amount'):
                # Only keep if the field itself is an aggregate call
                # Plain fields should never have these as aliases in SOQL
                if '(' not in field:
                    return field   # drop alias from plain field
            return field

        query = re.sub(
            r'(\b(?!SELECT\b)[\w_.]+(?:\([\w,\s_.]*\))?)\s+([a-zA-Z_]\w*)'
            r'(?=\s*(?:,|\bFROM\b))',
            _strip_plain_alias,
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX: ORDER BY aliases (legacy) ───────────────────────────
        query = re.sub(
            r'ORDER\s+BY\s+(cnt|count|total|total_leads|lead_count|value)\s+(DESC|ASC)',
            r'ORDER BY COUNT(Id) \2',
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX QV4: GROUP BY raw date fields ────────────────────────
        # DAY_ONLY() is ONLY valid inside GROUP BY on non-aggregate queries.
        # NEVER wrap ORDER BY date fields — that causes MALFORMED_QUERY.
        # Only wrap raw date fields in GROUP BY (not ORDER BY).
        _DATE_FIELDS = (
            'CreatedDate', 'TCL_Invoice_Date__c', 'EffectiveDate',
            'Visit_Planned_Date__c', 'Actual_Visit_Date__c',
            'CloseDate', 'ClosedDate', 'TCL_PO_Date__c',
        )
        _date_pattern = '|'.join(re.escape(f) for f in _DATE_FIELDS)

        # Only fix GROUP BY — never touch ORDER BY
        query = re.sub(
            rf'GROUP\s+BY\s+({_date_pattern})'
            rf'(\s+(?:ORDER|LIMIT|HAVING)|$)',
            r'GROUP BY DAY_ONLY(\1)\2',
            query,
            flags=re.IGNORECASE,
        )

        # ── REMOVE any DAY_ONLY() that ended up in ORDER BY ───────────────
        # Catches cases where previous validator versions or LLM added it.
        query = re.sub(
            r'ORDER\s+BY\s+DAY_ONLY\(\s*([\w_.]+)\s*\)',
            r'ORDER BY \1',
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX: Lead → Account region mapping ───────────────────────
        if re.search(r'\bFROM\s+Lead\b', query, flags=re.IGNORECASE):
            query = re.sub(
                r'Account\.TCL_Reporting_Region__c', 'Country',
                query,
                flags=re.IGNORECASE,
            )

        # ── AUTO-FIX: SUM(CASE WHEN ...) — not supported in SOQL ─────────
        query = re.sub(
            r',\s*SUM\s*\(\s*CASE\s*WHEN[^)]+\)\s+[\w_]+',
            '',
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX: Literal string aliases in SELECT ────────────────────
        query = re.sub(
            r"SELECT\s+'[^']+'\s+[\w_]+,\s*",
            'SELECT ',
            query,
            flags=re.IGNORECASE,
        )

        # ── AUTO-FIX: GROUP BY on multipicklist field ─────────────────────
        if re.search(
            r'GROUP\s+BY\s+[\w_]*TCL_Product_Business__c',
            query,
            flags=re.IGNORECASE,
        ):
            query = re.sub(
                r'\s+GROUP\s+BY\s+[\w_]*TCL_Product_Business__c(,\s*[\w_]+)*',
                '',
                query,
                flags=re.IGNORECASE,
            )
            query = re.sub(r'\s+ORDER\s+BY\s+.*$', '', query, flags=re.IGNORECASE)
            query = re.sub(
                r'SELECT\s+[\w_]*TCL_Product_Business__c\s*[\w_]*\s*,',
                'SELECT',
                query,
                flags=re.IGNORECASE,
            )

        return query


def validate_soql_query(query: str) -> Tuple[bool, str, Optional[str]]:
    """
    Main validation function for SOQL queries.

    Args:
        query: Raw SOQL query from LLM

    Returns:
        Tuple of (is_valid, cleaned_query, error_message)
    """
    cleaned = SOQLValidator.sanitize_query(query)
    is_valid, error = SOQLValidator.validate(cleaned)

    if not is_valid:
        logger.error(f"SOQL validation failed: {error}\nQuery: {query}")
        return False, cleaned, error

    logger.info(f"SOQL validation passed: {cleaned[:100]}...")
    return True, cleaned, None


def test_query_validator():
    """Test function to verify validator works correctly."""
    test_cases = [
        # Valid queries
        ("SELECT Id, Name FROM Lead WHERE Status = 'Open'", True),
        ("SELECT Id FROM TCL_Visit__c WHERE Visit_Planned_Date__c != null", True),
        ("SELECT COUNT(Id) cnt FROM TCL_Invoice__c GROUP BY TCL_Division_Description__c", True),
        ("SELECT Id, Status FROM Case WHERE Origin = 'Phone'", True),
        ("SELECT Account.Owner.Name, COUNT(Id) FROM Case GROUP BY Account.Owner.Name", True),
        ("SELECT Status__c, COUNT(Id) FROM TCL_CustomerOnBoarding__c GROUP BY Status__c", True),
        ("SELECT Name, Status FROM Campaign WHERE Type = 'Email'", True),
        ("SELECT Name, SurveyType FROM Survey WHERE CreatedDate = THIS_MONTH", True),
        # Plain ORDER BY date — must NOT get DAY_ONLY() wrapped
        ("SELECT CaseNumber, Status, CreatedDate FROM Case WHERE AccountId != null ORDER BY CreatedDate DESC LIMIT 20", True),
        ("SELECT Name, Actual_Visit_Date__c FROM TCL_Visit__c ORDER BY Actual_Visit_Date__c DESC LIMIT 10", True),
        ("SELECT OrderNumber, EffectiveDate FROM Order ORDER BY EffectiveDate DESC LIMIT 20", True),
        ("SELECT TCL_Invoice_Number__c, TCL_Invoice_Date__c FROM TCL_Invoice__c ORDER BY TCL_Invoice_Date__c DESC LIMIT 20", True),
        # Forecast date literals
        ("SELECT Name, CloseDate FROM Opportunity WHERE CloseDate = NEXT_MONTH ORDER BY CloseDate ASC LIMIT 20", True),
        ("SELECT Name FROM Opportunity WHERE CloseDate = NEXT_QUARTER", True),
        # CALENDAR aggregate aliases — must be preserved
        ("SELECT CALENDAR_MONTH(CreatedDate) month, COUNT(Id) value FROM Lead GROUP BY CALENDAR_MONTH(CreatedDate)", True),
        ("SELECT CALENDAR_YEAR(TCL_Invoice_Date__c) year, SUM(TCL_Invoice_Amount__c) total FROM TCL_Invoice__c GROUP BY CALENDAR_YEAR(TCL_Invoice_Date__c)", True),
        # QV2 — aggregation arithmetic stripped
        ("SELECT StageName, SUM(Amount * 0.3) weighted FROM Opportunity GROUP BY StageName", True),
        # QV1 — plain field alias stripped
        ("SELECT StageName stage, COUNT(Id) cnt FROM Opportunity GROUP BY StageName", True),
        # Aggregate relations
        (
            "SELECT Order.Account.TCL_Reporting_Region__c, SUM(TCL_Quantity_Metric_TON__c) "
            "FROM TCL_Order_Line_Items__c GROUP BY Order.Account.TCL_Reporting_Region__c",
            True,
        ),
        # QV3 — AVG on Satisfaction_score__c must be dropped cleanly, no GROUP BY injection
        (
            "SELECT COUNT(Id) visit_count, AVG(Satisfaction_score__c) avg_satisfaction, "
            "MIN(Actual_Visit_Date__c) first_visit, MAX(Actual_Visit_Date__c) last_visit "
            "FROM TCL_Visit__c WHERE Customer_Name__r.Name LIKE '%Hindustan Unilever Limited%'",
            True,
        ),
        (
            "SELECT COUNT(Id) VisitCount, AVG(Satisfaction_score__c) AvgSatisfaction, "
            "MAX(Visit_Date__c) LastVisitDate FROM TCL_Visit__c "
            "WHERE Customer_Name__r.Name LIKE '%Test Account%'",
            True,
        ),
        # QV3 edge case — AVG is the only field
        (
            "SELECT AVG(Satisfaction_score__c) FROM TCL_Visit__c WHERE Customer_Name__r.Name LIKE '%X%'",
            True,
        ),
        # Invalid queries
        ("DROP TABLE Lead", False),
        ("SELECT * FROM Lead", False),
        ("SELECT Id FROM Lead; DROP TABLE Account", False),
        ("SELECT Id FROM UnauthorizedObject__c", False),
    ]

    passed = 0
    failed = 0
    errors = []
    for q, expected in test_cases:
        is_valid, cleaned, err = validate_soql_query(q)
        ok = (is_valid == expected)
        status = "✅" if ok else "❌"
        if not ok:
            failed += 1
            errors.append((q, expected, is_valid, cleaned, err))
        else:
            passed += 1

        # Check DAY_ONLY not in ORDER BY for valid queries
        if is_valid and 'DAY_ONLY' in cleaned.upper() and 'ORDER BY' in cleaned.upper():
            ob_part = cleaned.upper().split('ORDER BY')[1]
            if 'DAY_ONLY' in ob_part:
                failed += 1
                passed -= 1
                errors.append((q, 'no DAY_ONLY in ORDER BY', 'DAY_ONLY found in ORDER BY', cleaned, ''))
                status = "❌"

        # QV3 specific check: GROUP BY must NOT appear inside a LIKE string
        if is_valid and 'GROUP BY' in cleaned.upper():
            like_matches = re.finditer(r"LIKE\s+'([^']*)'", cleaned, re.IGNORECASE)
            for m in like_matches:
                if 'GROUP BY' in m.group(1).upper():
                    failed += 1
                    passed -= 1
                    errors.append((q, 'GROUP BY not inside LIKE string', 'GROUP BY found inside LIKE string', cleaned, ''))
                    status = "❌"
                    break

        print(f"{status} {'valid' if expected else 'blocked'}: {q[:70]}")
        if cleaned != q and is_valid:
            print(f"   → cleaned: {cleaned[:120]}")

    print(f"\n{passed}/{passed+failed} passed")
    if errors:
        print("\nFailed cases:")
        for q, exp, got, cleaned, err in errors:
            print(f"  expected={exp}, got={got}: {q[:60]}")
            print(f"  cleaned: {cleaned[:120]}")
            print(f"  error:   {err}")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)
    test_query_validator()