# Instructions for AI coding agents (Codex, Claude, others)

Read and follow **`CLAUDE.md`** in this folder: it holds the project's north star, working principles,
git rules (commit trailer, no session links) and the handoff rule. They apply to every agent.

Then read `docs/03-status-and-next-steps.md` (start at "LATEST STATE") and continue from "Next actions".
After every completed step, update that file and push it to `main`.

Run the tests with `pytest -q -p no:cacheprovider` (CPU); never debug on paid H100s.
