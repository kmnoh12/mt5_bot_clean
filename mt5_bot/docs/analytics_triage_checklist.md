# Analytics Triage Checklist

Use this before another analytics-only patch so repeated Codex runs do not solve the same low-ROI problem twice.

- Confirm the question: bug, measurement gap, or reporting polish.
- Check whether an existing report/tool already answers it.
- Prefer read-only scripts and tests over live-runtime changes.
- Do not change broker APIs, order paths, credentials, or live gates for analytics-only work.
- Record commands run and the exact files inspected in the final handoff.
- If the result is inconclusive, leave a narrow next step instead of widening scope.
