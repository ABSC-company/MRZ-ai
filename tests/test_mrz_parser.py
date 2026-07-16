import unittest

from app.services.validation import parse_mrz


VALID_SAMPLES = [
    [
        "P<UKRSHARMAR<<ANDRII<<<<<<<<<<<<<<<<<<<<<<<<",
        "PU360255<2UKR8305044M29041131983050409911<86",
    ],
    [
        "P<FRARAYBAUD<GINES<<JULIEN<MICHEL<ALAIN<<<<<",
        "23FV102323FRA7508228M3401158<<<<<<<<<<<<<<06",
    ],
    [
        "P<RUSPOSOKINA<<NATALIA<<<<<<<<<<<<<<<<<<<<<<",
        "7320405111RUS6809283F2401302<<<<<<<<<<<<<<02",
    ],
    [
        "PNRUS3UMA3ENKO<<7RIQ<ALEKSANDROVI3<<<<<<<<<<",
        "6011369455RUS8007049M<<<<<<<2120207610009<62",
    ],
    [
        "PNRUSMINAKOV<<ANDREQ<LEONIDOVI3<<<<<<<<<<<<<",
        "1214225289RUS9203206M<<<<<<<1120512300002<50",
    ],
    [
        "P<UZBRAKHIMOV<<DIYOR<<<<<<<<<<<<<<<<<<<<<<<<",
        "AC06851141UZB0207049M28080825040702661003646",
    ],
    [
        "PDBLRVOITKEVICH<<MARIYA<<<<<<<<<<<<<<<<<<<<<",
        "PD00000004BLR8012296F27091344291280A112PB002",
    ],
    [
        "IUUZBAD3497356632805660530035<",
        "6605289M3305315UZBTAT<<<<<<<<2",
        "DAUDOV<<RAMIL<<<<<<<<<<<<<<<<<",
    ],
    [
        "IUUZBAD0974240352112056820024<",
        "0512217M3201071XXXUZB<<<<<<<<0",
        "DAUDOV<<RAIL<<<<<<<<<<<<<<<<<<",
    ],
    [
        "IDGEO13IN30572210001051644<<<<",
        "9105263M2512211GEO<<<<<<<<<<<0",
        "MAMEDOV<<IUKSEL<<<<<<<<<<<<<<<",
    ],
]


class MRZParserTests(unittest.TestCase):
    def test_supported_valid_samples(self):
        for lines in VALID_SAMPLES:
            with self.subTest(first_line=lines[0]):
                parsed = parse_mrz(lines)
                self.assertTrue(parsed["mrz_valid"], parsed)
                self.assertFalse(parsed["fallback_required"])

    def test_russian_domestic_profile(self):
        parsed = parse_mrz(VALID_SAMPLES[3])
        fields = parsed["parsed_fields"]
        self.assertEqual(fields["surname"], "ЧУМАЧЕНКО")
        self.assertEqual(fields["given_names"], "ЮРИЙ")
        self.assertEqual(fields["middle_name"], "АЛЕКСАНДРОВИЧ")
        self.assertEqual(fields["document_number"], "6012136945")
        self.assertEqual(fields["issue_date"], "2012-02-07")
        self.assertEqual(fields["issuing_authority_code"], "610009")
        self.assertIsNone(fields["expiry_date"])
        self.assertIsNone(fields["personal_number"])

    def test_uzbek_td1_personal_number(self):
        parsed = parse_mrz(VALID_SAMPLES[7])
        self.assertEqual(parsed["mrz_format"], "TD1")
        self.assertEqual(parsed["parsed_fields"]["personal_number"], "32805660530035")
        self.assertEqual(parsed["parsed_fields"]["birth_date"], "1966-05-28")

    def test_td3_numeric_optional_data_is_personal_number(self):
        ukrainian = parse_mrz(VALID_SAMPLES[0])
        self.assertEqual(ukrainian["parsed_fields"]["personal_number"], "1983050409911")

        french = parse_mrz(VALID_SAMPLES[1])
        self.assertIsNone(french["parsed_fields"]["personal_number"])

        filler_check = VALID_SAMPLES[1].copy()
        filler_check[1] = filler_check[1][:42] + "<" + filler_check[1][43]
        filler_only = parse_mrz(filler_check)
        self.assertTrue(filler_only["mrz_valid"])
        self.assertIsNone(filler_only["parsed_fields"]["personal_number"])

    def test_uzbek_xxx_nationality_uses_confirmed_optional_code(self):
        parsed = parse_mrz(VALID_SAMPLES[8])
        fields = parsed["parsed_fields"]
        self.assertEqual(fields["nationality_mrz"], "XXX")
        self.assertEqual(fields["nationality"], "UZB")

    def test_hungarian_ocr_confusion_is_not_silently_accepted(self):
        raw = [
            "I<HUNOOO017AE<2<<<<<<<<<<<<<<<",
            "7908150F2201041HUN<<<<<<<<<<<6",
            "MESZAROS<<BRIGITTA<ERZSEBET<<<",
        ]
        parsed = parse_mrz(raw)
        self.assertFalse(parsed["mrz_valid"])
        self.assertTrue(parsed["fallback_required"])

        corrected = raw.copy()
        corrected[0] = "I<HUN000017AE<2<<<<<<<<<<<<<<<"
        self.assertTrue(parse_mrz(corrected)["mrz_valid"])

    def test_incomplete_layout_requires_fallback(self):
        parsed = parse_mrz(["<<<<<<", "SURNAME<<NAME<<<<"])
        self.assertFalse(parsed["mrz_valid"])
        self.assertEqual(parsed["error"], "unsupported_or_incomplete_mrz_layout")

    def test_missing_mrz_requires_fallback(self):
        parsed = parse_mrz([])
        self.assertFalse(parsed["mrz_detected"])
        self.assertFalse(parsed["mrz_valid"])
        self.assertTrue(parsed["fallback_required"])
        self.assertEqual(parsed["error"], "mrz_not_detected")


if __name__ == "__main__":
    unittest.main()
