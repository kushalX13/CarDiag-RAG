"""Regression tests for live-demo answer formatting."""

import unittest

from carrecall_rag.rag_answer import generate_rag_answer


def _mk_campaign(
    campaign_number: str,
    component: str,
    text: str,
    snippet: str,
) -> dict:
    return {
        "campaign_number": campaign_number,
        "best_doc": {
            "campaign_number": campaign_number,
            "component": component,
            "text": text,
        },
        "evidence_snippets": [
            {
                "doc_id": f"{campaign_number}_0",
                "snippet": snippet,
            }
        ],
    }


class TestDemoOutputFormat(unittest.TestCase):
    def test_demo_output_has_required_sections(self) -> None:
        query = "airbag warning light ORC module clock spring may disable air bags"
        vehicle = "Jeep Grand Cherokee"
        campaigns = [
            _mk_campaign(
                "18V332000",
                "Occupant Restraint Controller (ORC)",
                "The ORC module may disable airbag deployment and illuminate the warning light.",
                "Clock spring and ORC faults may disable the airbags.",
            ),
            _mk_campaign(
                "17V001000",
                "Brake hose",
                "Brake hose wear can reduce brake function.",
                "This recall is related to braking performance.",
            ),
        ]

        out = generate_rag_answer(query, campaigns, top_k=3, vehicle=vehicle)

        self.assertIn("Possible Recall-Related Issue", out)
        self.assertIn("Vehicle:\nJeep Grand Cherokee", out)
        self.assertIn("Query:\nairbag warning light ORC module clock spring may disable air bags", out)
        self.assertIn("Best Match:\n18V332000 — Occupant Restraint Controller (ORC)", out)
        self.assertIn("Why it matches:", out)
        self.assertIn("Other relevant recall candidates:", out)
        self.assertIn("Potential safety risk:", out)
        self.assertIn("Recommended next step:", out)

    def test_demo_output_labels_weak_alternates_lower_confidence(self) -> None:
        query = "airbag warning light ORC module clock spring may disable air bags"
        campaigns = [
            _mk_campaign(
                "18V332000",
                "Occupant Restraint Controller (ORC)",
                "The ORC module may disable airbag deployment and illuminate the warning light.",
                "Clock spring and ORC faults may disable the airbags.",
            ),
            _mk_campaign(
                "14V643000",
                "Cruise control switch",
                "Cruise control wiring may short and overheat.",
                "This campaign concerns cruise control wiring.",
            ),
        ]

        out = generate_rag_answer(query, campaigns, top_k=3, vehicle="Jeep Grand Cherokee")

        self.assertIn("Other relevant recall candidates:", out)
        self.assertIn("- 14V643000 — Cruise control switch (lower-confidence)", out)

    def test_demo_output_no_candidate_fallback_is_presentation_safe(self) -> None:
        query = "unknown issue text"
        out = generate_rag_answer(query, [], top_k=3, vehicle="Ford F-150")

        self.assertIn("Possible Recall-Related Issue", out)
        self.assertIn("Vehicle:\nFord F-150", out)
        self.assertIn("Query:\nunknown issue text", out)
        self.assertIn("Why it matches:", out)
        self.assertNotIn("Best Match:", out)
        self.assertIn("Potential safety risk:", out)
        self.assertIn("Recommended next step:", out)
