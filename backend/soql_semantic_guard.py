import re
from typing import Tuple, Optional
from query_validator import validate_soql_query


SQL_ONLY_PATTERNS = [
    r'\bCASE\b',
    r'\bWHEN\b',
    r'\bTHEN\b',
    r'\bELSE\b',
    r'\bEND\b',
    r'\bJOIN\b',
]

DATETIME_FIELDS = [
    'CREATEDDATE',
    'LASTMODIFIEDDATE',
    'CONVERTEDDATE'
]

ALLOWED_FIELDS = {
    'LEAD': [
        'ID',
        'LEADSOURCE',
        'STATUS',
        'CREATEDDATE',
        'ISCONVERTED',
        'CONVERTEDDATE',
        'TCL_REPORTING_REGION__C',
        'TCL_DROPPED__C'
    ]
}


def semantic_guard(query: str) -> Tuple[bool, str, Optional[str]]:
    """
    Wraps existing validator with SOQL semantic validation
    WITHOUT modifying original validator
    """

    # Step 1 — run existing validator
    is_valid, cleaned, error = validate_soql_query(query)

    if not is_valid:
        return False, cleaned, error

    query_upper = cleaned.upper()

    # Step 2 — block SQL-only constructs
    for pattern in SQL_ONLY_PATTERNS:
        if re.search(pattern, query_upper):
            return False, cleaned, "SQL-style construct not allowed in SOQL"

    # Step 3 — datetime group by guard
    group_match = re.search(r'GROUP BY (.+?)(?: ORDER BY| LIMIT|$)', query_upper)
    if group_match:
        group_fields = group_match.group(1)
        for dt in DATETIME_FIELDS:
            if re.search(rf'\b{dt}\b', group_fields):
                if not re.search(rf'(CALENDAR_|DAY_ONLY|HOUR_IN_DAY).*{dt}', query_upper):
                    return False, cleaned, f"Cannot group by raw datetime field {dt}"

    # Step 4 — minimal field validation
    from_match = re.search(r'FROM\s+(\w+)', query_upper)
    select_match = re.search(r'SELECT (.+?) FROM', query_upper)

    if from_match and select_match:
        object_name = from_match.group(1)
        fields = select_match.group(1)

        if object_name in ALLOWED_FIELDS:
            for f in re.split(r',\s*', fields):

                # Ignore aggregate expressions and date functions
                if re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|CALENDAR_|DAY_ONLY|HOUR_IN_DAY)\s*\(', f):
                    continue

                # Remove alias
                f_no_alias = re.split(r'\s+AS\s+|\s+', f, flags=re.IGNORECASE)[0]

                # Remove function wrappers
                f_clean = re.sub(r'\(.*?\)', '', f_no_alias).strip()

                if '.' not in f_clean and f_clean not in ALLOWED_FIELDS.get(object_name, []):
                    return False, cleaned, f"Field not allowed: {f_clean}"

    # Step 5 — conditional aggregation guard
    if "SUM(" in query_upper and "ISCONVERTED" in query_upper:
        return False, cleaned, "Conditional aggregation not supported in SOQL"

    return True, cleaned, None