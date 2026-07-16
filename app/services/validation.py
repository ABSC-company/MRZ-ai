from __future__ import annotations

from datetime import date, datetime
from typing import Any


MRZ_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
MRZ_VALUES = {str(value): value for value in range(10)}
MRZ_VALUES.update({chr(ord("A") + index): 10 + index for index in range(26)})
MRZ_VALUES["<"] = 0
MRZ_WEIGHTS = (7, 3, 1)

RUS_CLER_TO_CYRILLIC = str.maketrans(
    {
        "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "E": "Е",
        "2": "Ё", "J": "Ж", "Z": "З", "I": "И", "Q": "Й", "K": "К",
        "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р",
        "S": "С", "T": "Т", "U": "У", "F": "Ф", "H": "Х", "C": "Ц",
        "3": "Ч", "4": "Ш", "W": "Щ", "X": "Ъ", "Y": "Ы", "9": "Ь",
        "6": "Э", "7": "Ю", "8": "Я",
    }
)


def sanitize_mrz_lines(texts: list[str]) -> list[str]:
    lines: list[str] = []
    for text in texts:
        clean = "".join(character for character in str(text).upper() if character in MRZ_CHARS)
        if clean:
            lines.append(clean)
    return lines


def check_digit(value: str) -> int:
    return sum(
        MRZ_VALUES.get(character, 0) * MRZ_WEIGHTS[index % len(MRZ_WEIGHTS)]
        for index, character in enumerate(value)
    ) % 10


def check_field(value: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(value) == int(digit)


def _empty_checked_field(value: str, digit: str) -> bool:
    return bool(value) and set(value) == {"<"} and digit == "<"


def _clean_value(value: str) -> str | None:
    clean = value.replace("<", " ").strip()
    clean = " ".join(clean.split())
    return clean or None


def _resolve_birth_date(raw: str, today: date | None = None) -> str | None:
    if len(raw) != 6 or not raw.isdigit():
        return None
    today = today or date.today()
    candidates: list[date] = []
    for century in (today.year // 100 - 1, today.year // 100):
        try:
            candidate = datetime.strptime(f"{century}{raw}", "%Y%m%d").date()
        except ValueError:
            continue
        age = today.year - candidate.year - ((today.month, today.day) < (candidate.month, candidate.day))
        if candidate <= today and 0 <= age <= 120:
            candidates.append(candidate)
    return max(candidates).isoformat() if candidates else None


def _resolve_expiry_date(raw: str, today: date | None = None) -> str | None:
    if len(raw) != 6 or not raw.isdigit():
        return None
    today = today or date.today()
    candidates: list[date] = []
    for century in (today.year // 100 - 1, today.year // 100, today.year // 100 + 1):
        try:
            candidates.append(datetime.strptime(f"{century}{raw}", "%Y%m%d").date())
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item - today).days)).isoformat()


def _resolve_issue_date(raw: str, today: date | None = None) -> str | None:
    return _resolve_birth_date(raw, today=today)


def _decode_rus_name(value: str) -> str | None:
    clean = _clean_value(value)
    return clean.translate(RUS_CLER_TO_CYRILLIC) if clean else None


def _parse_names(raw: str, rus_domestic: bool = False) -> dict[str, Any]:
    surname_raw, separator, secondary_raw = raw.partition("<<")
    surname = _decode_rus_name(surname_raw) if rus_domestic else _clean_value(surname_raw)
    secondary = [item for item in secondary_raw.split("<") if item] if separator else []

    if rus_domestic:
        decoded = [_decode_rus_name(item) for item in secondary]
        decoded = [item for item in decoded if item]
        given_names = decoded[0] if decoded else None
        middle_name = " ".join(decoded[1:]) or None
        all_given_names = decoded
    else:
        all_given_names = secondary
        given_names = " ".join(secondary) or None
        middle_name = None

    return {
        "surname": surname,
        "given_names": given_names,
        "given_names_all": all_given_names,
        "middle_name": middle_name,
    }


def _detect_format(lines: list[str]) -> str | None:
    lengths = [len(line) for line in lines]
    if len(lines) == 3 and lengths == [30, 30, 30]:
        return "TD1"
    if len(lines) == 2 and lengths == [44, 44]:
        return "MRVA" if lines[0].startswith("V") else "TD3"
    if len(lines) == 2 and lengths == [36, 36]:
        return "MRVB" if lines[0].startswith("V") else "TD2"
    return None


def _base_fields(document_type: str, issuer: str, names_raw: str) -> dict[str, Any]:
    rus_domestic = document_type == "PN" and issuer == "RUS"
    fields: dict[str, Any] = {
        "document_type": document_type.replace("<", "") or None,
        "issuing_state_code": _clean_value(issuer),
    }
    fields.update(_parse_names(names_raw, rus_domestic=rus_domestic))
    return fields


def _parse_td3(lines: list[str]) -> tuple[dict[str, Any], dict[str, bool | None]]:
    first, second = lines
    document_type = first[0:2]
    issuer = first[2:5]
    fields = _base_fields(document_type, issuer, first[5:44])
    optional_data = second[28:42]
    expiry_raw = second[21:27]
    expiry_absent = _empty_checked_field(expiry_raw, second[27])

    fields.update(
        {
            "document_number": _clean_value(second[0:9]),
            "nationality": _clean_value(second[10:13]),
            "birth_date_raw": second[13:19],
            "birth_date": _resolve_birth_date(second[13:19]),
            "sex": None if second[20] == "<" else second[20],
            "expiry_date_raw": None if expiry_absent else second[21:27],
            "expiry_date": None if expiry_absent else _resolve_expiry_date(second[21:27]),
            "personal_number": None,
            "optional_data_raw": optional_data,
        }
    )
    personal_number = optional_data.rstrip("<")
    is_rus_domestic = document_type == "PN" and issuer == "RUS"
    if not is_rus_domestic and personal_number.isdigit() and len(personal_number) in {13, 14}:
        fields["personal_number"] = personal_number

    optional_empty = set(optional_data) == {"<"}
    optional_check = None if optional_empty and second[42] == "<" else check_field(optional_data, second[42])
    checks: dict[str, bool | None] = {
        "document_number": check_field(second[0:9], second[9]),
        "birth_date": check_field(second[13:19], second[19]),
        "expiry_date": None if expiry_absent else check_field(expiry_raw, second[27]),
        "optional_data": optional_check,
        "composite": check_field(second[0:10] + second[13:20] + second[21:43], second[43]),
    }

    if document_type == "PN" and issuer == "RUS":
        last_series_digit = optional_data[0]
        if last_series_digit.isdigit() and second[0:9].isdigit():
            fields["document_number_mrz"] = second[0:9]
            fields["document_number"] = second[0:3] + last_series_digit + second[3:9]
        fields["issue_date"] = _resolve_issue_date(optional_data[1:7])
        fields["issuing_authority_code"] = _clean_value(optional_data[7:13])

    return fields, checks


def _parse_td2(lines: list[str]) -> tuple[dict[str, Any], dict[str, bool | None]]:
    first, second = lines
    fields = _base_fields(first[0:2], first[2:5], first[5:36])
    fields.update(
        {
            "document_number": _clean_value(second[0:9]),
            "nationality": _clean_value(second[10:13]),
            "birth_date_raw": second[13:19],
            "birth_date": _resolve_birth_date(second[13:19]),
            "sex": None if second[20] == "<" else second[20],
            "expiry_date_raw": second[21:27],
            "expiry_date": _resolve_expiry_date(second[21:27]),
            "personal_number": None,
            "optional_data_raw": second[28:35],
        }
    )
    checks: dict[str, bool | None] = {
        "document_number": check_field(second[0:9], second[9]),
        "birth_date": check_field(second[13:19], second[19]),
        "expiry_date": check_field(second[21:27], second[27]),
        "composite": check_field(second[0:10] + second[13:20] + second[21:35], second[35]),
    }
    return fields, checks


def _parse_td1(lines: list[str]) -> tuple[dict[str, Any], dict[str, bool | None]]:
    first, second, third = lines
    fields = _base_fields(first[0:2], first[2:5], third)
    optional_data_1 = first[15:30]
    optional_data_2 = second[18:29]
    fields.update(
        {
            "document_number": _clean_value(first[5:14]),
            "nationality": _clean_value(second[15:18]),
            "birth_date_raw": second[0:6],
            "birth_date": _resolve_birth_date(second[0:6]),
            "sex": None if second[7] == "<" else second[7],
            "expiry_date_raw": second[8:14],
            "expiry_date": _resolve_expiry_date(second[8:14]),
            "personal_number": None,
            "optional_data_raw": optional_data_1 + optional_data_2,
            "optional_data_1_raw": optional_data_1,
            "optional_data_2_raw": optional_data_2,
        }
    )
    if first[0:2] == "IU" and first[2:5] == "UZB":
        candidate = optional_data_1.rstrip("<")
        if len(candidate) == 14 and candidate.isdigit():
            fields["personal_number"] = candidate
        optional_nationality = optional_data_2[0:3]
        if fields["nationality"] == "XXX" and optional_nationality == "UZB":
            fields["nationality_mrz"] = "XXX"
            fields["nationality"] = "UZB"

    checks: dict[str, bool | None] = {
        "document_number": check_field(first[5:14], first[14]),
        "birth_date": check_field(second[0:6], second[6]),
        "expiry_date": check_field(second[8:14], second[14]),
        "composite": check_field(
            first[5:30] + second[0:7] + second[8:15] + second[18:29], second[29]
        ),
    }
    return fields, checks


def _parse_visa(lines: list[str], line_length: int) -> tuple[dict[str, Any], dict[str, bool | None]]:
    first, second = lines
    fields = _base_fields(first[0:2], first[2:5], first[5:line_length])
    fields.update(
        {
            "document_number": _clean_value(second[0:9]),
            "nationality": _clean_value(second[10:13]),
            "birth_date_raw": second[13:19],
            "birth_date": _resolve_birth_date(second[13:19]),
            "sex": None if second[20] == "<" else second[20],
            "expiry_date_raw": second[21:27],
            "expiry_date": _resolve_expiry_date(second[21:27]),
            "personal_number": None,
            "optional_data_raw": second[28:line_length],
        }
    )
    checks: dict[str, bool | None] = {
        "document_number": check_field(second[0:9], second[9]),
        "birth_date": check_field(second[13:19], second[19]),
        "expiry_date": check_field(second[21:27], second[27]),
    }
    return fields, checks


def parse_mrz(texts: list[str]) -> dict[str, Any]:
    lines = sanitize_mrz_lines(texts)
    mrz_format = _detect_format(lines)
    base: dict[str, Any] = {
        "mrz_detected": bool(lines),
        "mrz_format": mrz_format,
        "mrz_valid": False,
        "mrz_text_raw": [str(text) for text in texts],
        "mrz_text_normalized": lines,
        "parsed_fields": {},
        "validation": {},
        "validation_score": 0.0,
        "fallback_required": True,
    }
    if mrz_format is None:
        base["error"] = "unsupported_or_incomplete_mrz_layout" if lines else "mrz_not_detected"
        return base

    if mrz_format == "TD1":
        fields, checks = _parse_td1(lines)
    elif mrz_format == "TD2":
        fields, checks = _parse_td2(lines)
    elif mrz_format == "TD3":
        fields, checks = _parse_td3(lines)
    elif mrz_format == "MRVA":
        fields, checks = _parse_visa(lines, 44)
    else:
        fields, checks = _parse_visa(lines, 36)

    applicable = [value for value in checks.values() if value is not None]
    fully_valid = bool(applicable) and all(value is True for value in applicable)

    base.update(
        {
            "mrz_valid": fully_valid,
            "parsed_fields": fields,
            "validation": checks,
            "validation_score": round(sum(value is True for value in applicable) / max(len(applicable), 1), 4),
            "fallback_required": not fully_valid,
        }
    )
    return base


def mrz_fields_for_api(parsed: dict[str, Any]) -> dict[str, Any]:
    fields = parsed.get("parsed_fields", {})
    return {
        "birth_date": fields.get("birth_date"),
        "expiry_date": fields.get("expiry_date"),
        "surname": fields.get("surname"),
        "given_names": fields.get("given_names"),
        "middle_name": fields.get("middle_name"),
        "document_number": fields.get("document_number"),
        "personal_number": fields.get("personal_number"),
        "sex": fields.get("sex"),
        "nationality": fields.get("nationality"),
    }
