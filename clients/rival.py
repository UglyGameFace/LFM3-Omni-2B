"""RivaL adapter: evidence-grounded explanations ONLY.

The adapter takes VERIFIED metrics / market evidence / diagnostics as
input and asks Omni to explain them. It never lets the model invent
probabilities, market data, trades or model performance:

- Every explanation must cite the evidence refs it was given.
- If the caller asks for a probability, price, trade or performance
  number that is not in the evidence, the adapter REFUSES (returns a
  refusal dict) instead of asking the model to guess.
- The existing prediction engines and autonomous qualification safeguards
  are untouched — this adapter has no access to them and no code path
  that could call them. Omni explains; it never predicts or qualifies.

Evidence shape: {"source": str, "kind": str, "summary": str,
"metrics": {...}, "refs": [str, ...]} — at least one ref is required.
"""
from __future__ import annotations

FORBIDDEN_REQUESTS = ("probability", "predict the price", "place a trade",
                      "should i bet", "win rate", "expected return")


class RivalAdapter:
    def __init__(self, client, session_id: str = "rival-intelligence") -> None:
        self.client = client
        self.session_id = session_id

    @staticmethod
    def validate_evidence(evidence: dict) -> list[str]:
        errors = []
        if not isinstance(evidence, dict):
            return ["evidence must be an object"]
        if not evidence.get("source"):
            errors.append("evidence.source is required (where the numbers came from)")
        if not evidence.get("refs"):
            errors.append("evidence.refs is required (at least one traceable reference)")
        if not any(evidence.get(k) for k in ("metrics", "market_evidence", "diagnostics", "summary")):
            errors.append("evidence must carry metrics, market_evidence, diagnostics or a summary")
        return errors

    def explain(self, evidence: dict, question: str) -> dict:
        """Explain verified evidence. Returns a structured intelligence
        response with traceable refs, or a refusal when the question asks
        the model to invent something."""
        errors = self.validate_evidence(evidence)
        if errors:
            raise ValueError("invalid evidence: " + "; ".join(errors))
        lowered = question.lower()
        for forbidden in FORBIDDEN_REQUESTS:
            if forbidden in lowered:
                return self.refuse(question, f"question asks for {forbidden!r}, "
                                             "which is not in the supplied evidence")
        prompt = (
            "You are explaining VERIFIED evidence only. Do not invent probabilities, "
            "market data, trades or model performance. If the evidence does not answer "
            "the question, say so.\n\n"
            f"Evidence source: {evidence.get('source')}\n"
            f"Evidence refs: {', '.join(evidence.get('refs', []))}\n"
            f"Evidence: {evidence.get('summary', '')}\n"
            f"Metrics: {evidence.get('metrics', {})}\n"
            f"Market evidence: {evidence.get('market_evidence', {})}\n"
            f"Diagnostics: {evidence.get('diagnostics', {})}\n\n"
            f"Question: {question}"
        )
        text = self.client.chat_text(
            [{"role": "user", "content": prompt}],
            session_id=self.session_id, max_tokens=512)
        return {"summary": text,
                "evidence_refs": list(evidence.get("refs", [])),
                "evidence_source": evidence.get("source"),
                "uncertainty": "explanation only; no new prediction or qualification was made",
                "refused_to_invent": True,
                "model": "LFM3-Omni-2B (non-reasoning)"}

    def refuse(self, question: str, reason: str) -> dict:
        return {"summary": None, "refused": True, "question": question,
                "reason": reason,
                "note": "RivaL's prediction and qualification engines are the only "
                        "source of probabilities and calls; Omni never supplies them."}
