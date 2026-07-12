# Getting Started — Agent Stress-Test

A one-time setup checklist to complete before beginning Phase 1. Work through it top to bottom. Once done, you won't need this file again — day-to-day building runs off the build plan and CLAUDE.md.

---

## 1. Accounts, Keys & Budget

- [ ] Anthropic API key ready.
- [ ] OpenAI API key ready.
- [ ] Confirm API budget / rate limits so you don't get surprised mid-build.
- [ ] (Optional, only for the Phase 11 stretch) Confirm GPU access for a trained failure-classifier.
- [ ] Keys will go in a local `.env` file, referenced by config — never hardcoded, never committed.

---

## 2. Local Environment

- [ ] Python 3.11+ installed and confirmed (`python --version`).
- [ ] Claude Code installed and working.
- [ ] Editor of choice ready (Cursor, VS Code, etc.).
- [ ] Create a fresh, empty git repository. Start version control from the first commit so each phase is a clean checkpoint.
- [ ] Create a `.env` file (and add it to `.gitignore` immediately) for `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

---

## 3. Real-World Target Agent (for the "works on any agent" demo)

- [ ] Primary: clone `rulyone/Simple-ReAct-Agent`.
  - [ ] Verify its license is permissive (MIT/Apache) before using it in an org demo.
  - [ ] Install Ollama and pull Llama 3.2 3B (~3 GB, one-time). Confirm it runs.
  - [ ] Run the agent once by hand to confirm it works before wiring it in (this happens in Phase 2, but confirm the environment now).
- [ ] Backup (only if Ollama is a problem): clone `mattambrogi/agent-implementation` (needs an OpenAI key).
- [ ] Note: your own bundled sample agent is the primary demo target; the real agent is the credibility add-on. You do NOT need the real agent working to start Phase 1.

---

## 4. Bundled Sample Agent — rough sketch

Have a starting sketch ready before Phase 2 (doesn't need to be final):

- [ ] A short system prompt for a general tool-calling / ReAct-style agent.
- [ ] 2-3 tools (name + what each does).
- [ ] 3-4 stated rules the agent must follow (these become what the judge checks against).

---

## 5. Skills (optional but recommended)

Install into Claude Code / Cursor before Phase 1 so they're active from the start. **Read each SKILL.md before installing — treat third-party skills as untrusted code.**

- [ ] A pytest / Python-testing skill (supports the per-phase test workflow).
- [ ] A litellm skill (helps write the provider layer correctly).
- [ ] An LLM-evaluation skill (supports the judge).
- [ ] A hexagonal / clean-architecture skill (protects the layering).
- [ ] (Later) frontend-design skill for the Phase 10 dashboard (styles the htmx + Alpine.js + Tailwind CSS frontend); docx/pptx for write-ups and the demo deck.

---

## 6. Documents On Hand & Where They Go

At the repo root (Claude Code reads these):
- [ ] `CLAUDE.md` — guidance file; must be at the root with this exact name so Claude Code reads it automatically.
- [ ] `agent_stress_test_build_plan.md` — the phased build plan; at the root, referenced by the phase prompts.

In `docs/` (for you):
- [ ] `GETTING_STARTED.md` — this checklist.

Kept separately (not in the repo):
- [ ] `Agent_Stress_Test.docx` — the org submission doc (keep consistent as the project evolves).

---

## 7. Decide Before Phase 1

- [ ] Package name: the plan uses `src/ast/`, but `ast` shadows Python's standard-library module. Decide now whether to rename the package (e.g. `agent_stress_test`) to avoid confusion. Recommended: rename.

---

## Starting Sequence (do in this order)

1. Complete sections 1-2 (env, keys, empty git repo, `.env` + `.gitignore`).
2. Install the skills from section 5 (after reading them).
3. Place `CLAUDE.md` and `agent_stress_test_build_plan.md` at the repo root; put `GETTING_STARTED.md` and `PHASE_PROMPTS.txt` in a `docs/` folder. (Phase 0 will also set up `docs/` if you haven't.)
4. Have the sample-agent sketch (section 4) ready.
5. Open Claude Code, and paste the **Phase 0** block from `docs/PHASE_PROMPTS.txt` to align it and scaffold the skeleton.
6. Then work through the phase prompts one at a time, reviewing and committing after each.
7. Set up the real-world target agent (section 3) any time before Phase 2 — it's not needed for Phase 1.

---

## First-Week Reminders

- Tests gate progress: no new phase on top of failing tests.
- The main test suite must run fully offline (fake provider), no live API needed.
- Don't add dependencies outside the approved list (litellm, pydantic, pytest, rich, hdbscan/sklearn, sentence-transformers).
- Don't over-engineer: no ports/adapters for things with only one implementation.
- Commit at the end of every phase as a clean checkpoint.
- Phase 9 (CLI product) is your safe stopping point; Phase 10 (dashboard) makes it demo-shine and lets you trigger/monitor runs from the browser too; Phase 11 (MCTS) is bonus only.
