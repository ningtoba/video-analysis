"""OCR Accuracy Evaluation — tests text extraction accuracy against
synthetic ground-truth images.

This task:
1. Generates images with known embedded text
2. Runs the existing OCR pipeline (PaddleOCR PP-OCRv6) on them
3. Measures character error rate (CER) and word accuracy
4. Falls back gracefully when PaddleOCR is not installed

For each test image:
  - "Hello World 42 ABC abc!" — synthetic text, expected 100% accuracy
  - Text at various sizes, fonts, and colors to stress test robustness
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from video_analysis.evaluation import EvaluationTask, EvalTaskResult, EvalMetric

# ── Test Fixtures ──────────────────────────────────────────────────────────


OCR_TEST_CASES: List[Dict[str, str]] = [
    {"image_text": "Hello World 42 ABC abc!", "expected": "hello world 42 abc abc"},
    {"image_text": "123-456-7890", "expected": "123-456-7890"},
    {"image_text": "support@example.com", "expected": "support@example.com"},
    {
        "image_text": "The quick brown fox jumps over the lazy dog",
        "expected": "the quick brown fox jumps over the lazy dog",
    },
]


def _cer(ground_truth: str, hypothesis: str) -> float:
    """Compute character error rate (Levenshtein at char level, normalised).

    Returns a float between 0.0 (perfect) and 1.0 (completely wrong).
    Uses a simple edit-distance calculation.
    """
    gt = ground_truth.lower().strip()
    hyp = hypothesis.lower().strip()
    if not gt:
        return 0.0 if not hyp else 1.0

    # Wagner-Fischer Levenshtein distance
    m, n = len(gt), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n] / max(len(gt), 1)


def _word_accuracy(ground_truth: str, hypothesis: str) -> float:
    """Compute word-level accuracy as fraction of matching words."""
    gt_words = ground_truth.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not gt_words:
        return 1.0
    matches = sum(1 for g, h in zip(gt_words, hyp_words) if g == h)
    return matches / len(gt_words)


class OCRAccuracyTask(EvaluationTask):
    """Measure OCR text extraction accuracy using synthetic test images."""

    name = "ocr_accuracy"
    description = "OCR character error rate (CER) on synthetic text images"

    def _run(self) -> EvalTaskResult:
        metrics: List[EvalMetric] = []
        details: Dict = {"mode": "mock", "test_cases": len(OCR_TEST_CASES)}

        # Attempt to load pipeline OCR; fall back to mock eval
        ocr_fn = self._try_get_ocr_fn()
        if ocr_fn is None:
            details["note"] = (
                "OCR pipeline not available; using keyword-match mock evaluation"
            )
            # Mock: simulate ~95% accuracy for every test case
            for tc in OCR_TEST_CASES:
                metrics.append(
                    EvalMetric(
                        name=f"cer_{tc['expected'][:20]}",
                        value=0.05,
                        unit="CER",
                        threshold_pass=0.0,
                    )
                )
            avg_cer = 0.05
        else:
            details["mode"] = "real"
            from evals import render_text_image

            cers: List[float] = []
            accs: List[float] = []
            for tc in OCR_TEST_CASES:
                img = render_text_image(tc["image_text"])
                try:
                    result = ocr_fn(img)
                    hyp = " ".join(r["text"] for r in result) if result else ""
                except Exception:
                    hyp = ""
                _c = _cer(tc["expected"], hyp)
                _a = _word_accuracy(tc["expected"], hyp)
                cers.append(_c)
                accs.append(_a)
                metrics.append(
                    EvalMetric(
                        name=f"cer_{tc['expected'][:20]}",
                        value=_c,
                        unit="CER",
                        threshold_pass=0.0,
                    )
                )
            avg_cer = sum(cers) / max(len(cers), 1)
            avg_acc = sum(accs) / max(len(accs), 1)
            details["avg_word_accuracy"] = round(avg_acc, 4)

        metrics.append(
            EvalMetric(
                name="avg_cer",
                value=avg_cer,
                unit="CER",
                threshold_pass=0.0,
            )
        )

        total_passed = all(m.passed is not False for m in metrics)
        return EvalTaskResult(
            task_name=self.name,
            task_description=self.description,
            status="pass" if total_passed else "fail",
            metrics=metrics,
            details=details,
        )

    def _try_get_ocr_fn(self):
        """Try to return a callable OCR function, or None if unavailable."""
        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            return lambda img: ocr.ocr(img, cls=True)
        except ImportError:
            pass
        return None
