import unittest

import pandas as pd

import config
from src.harmonize import (care_module_mask, cohort_counts, complete_followup_mask,
                           derive_neonatal_outcome, derive_skilled_attendant,
                           recode_anc_4plus)


class HarmonizeBoundaryTests(unittest.TestCase):
    def test_neonatal_boundary_is_days_0_through_27(self):
        raw = pd.DataFrame({"b5": [0, 0, 0, 0], "b6": [100, 127, 128, 129]})
        self.assertEqual(derive_neonatal_outcome(raw).tolist(), [1, 1, 0, 0])

    def test_invalid_age_at_death_is_unclassifiable_but_survivor_is_not(self):
        raw = pd.DataFrame({
            "b5": [0, 0, 0, 0, 0, 0, 1],
            "b6": [99, 199, 299, 399, 999, None, None],
        })
        got = derive_neonatal_outcome(raw)
        self.assertTrue(got.iloc[:6].isna().all())
        self.assertEqual(got.iloc[6], 0)

    def test_exact_non_neonatal_day_codes_remain_classifiable(self):
        # DHS b6 uses 1xx for reported days; days 28-90 are non-neonatal,
        # whereas last-two-digit values above 90 are special responses.
        raw = pd.DataFrame({"b5": [0, 0, 0, 0], "b6": [128, 129, 135, 190]})
        self.assertEqual(derive_neonatal_outcome(raw).tolist(), [0, 0, 0, 0])

    def test_care_module_uses_design_not_item_response(self):
        raw = pd.DataFrame({
            "survey_year": [2014, 2014, 2022, 2022, 2022],
            "bidx": [1, 2, 1, 1, 2],
            "sqtype": [None, None, 1, 2, 1],
            "m15": [None, 21, None, 21, 21],
        })
        self.assertEqual(care_module_mask(raw).tolist(),
                         [True, False, True, False, False])

    def test_birth_month_survivors_wait_until_followup_is_complete(self):
        ages = pd.Series([0, 1, 27, 35, 36, None])
        self.assertEqual(complete_followup_mask(ages).tolist(),
                         [False, False, True, True, False, False])
        self.assertEqual(complete_followup_mask(ages, min_months=1).tolist(),
                         [False, True, True, True, False, False])

    def test_per_round_cohort_arithmetic(self):
        df = pd.DataFrame({
            config.YEAR_COL: [2011, 2011, 2014, 2014, 2014, 2022],
            config.TARGET: [0, 1, 0, 0, 1, 1],
        })
        got = cohort_counts(df)
        self.assertEqual(got.to_dict("index"), {
            2011: {"n": 2, "deaths": 1},
            2014: {"n": 3, "deaths": 1},
            2022: {"n": 1, "deaths": 1},
        })

    def test_care_recodes_preserve_missing_and_include_all_skilled_cadres(self):
        anc = recode_anc_4plus(pd.Series([None, 0, 3, 4]))
        self.assertTrue(pd.isna(anc.iloc[0]))
        self.assertEqual(anc.iloc[1:].tolist(), [0, 0, 1])
        raw = pd.DataFrame({
            "m3a": [None, 0, 0], "m3b": [None, 0, 0],
            "m3c": [None, 1, 0], "m3d": [None, 0, 0], "m3e": [None, 0, 1],
        })
        got = derive_skilled_attendant(raw)
        self.assertTrue(pd.isna(got.iloc[0]))
        self.assertEqual(got.iloc[1:].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
