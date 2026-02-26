# Code Evaluator V2 — Multi-Agent Evaluation Framework

Upgrade from V1's single-LLM judge to a **multi-agent, multi-model** evaluation pipeline with hallucination auditing and deterministic-first scoring.

## Architecture

```
CLI (cli.py)
  ↓
Orchestrator (core/orchestrator.py)
  ↓
Deterministic Engine (core/deterministic_engine.py)
  │  sandbox → test runner → static analysis → rubric scoring
  ↓
LLM Agent Layer
  ├── Test Designer Agent    (agents/test_designer.py)
  ├── Code Reviewer Agent    (agents/code_reviewer.py)
  ├── Complexity Analyst     (agents/complexity_analyst.py)
  ↓
Consensus Agent              (agents/consensus_agent.py)
  ↓
Hallucination Auditor        (safety/hallucination_auditor.py)
  ↓
Score Blending → Final Score
  ↓
Report Generator → evaluation_report.md + results.json
```

## Agent Responsibilities

| Agent | Role |
|-------|------|
| **Test Designer** | Reviews submission, suggests edge-case tests, scores correctness/edge-cases |
| **Code Reviewer** | Evaluates style, readability, maintainability, best practices |
| **Complexity Analyst** | Estimates Big-O time/space complexity, flags inefficiency |
| **Consensus Agent** | Merges agent scores, resolves disagreements, produces unified LLM score |

## Model Routing

Configured in `config.yaml` under `models:`. The system:

1. Queries HuggingFace model metadata (`/api/models/{id}?expand=inferenceProviderMapping`)
2. Detects supported tasks (`conversational` → chat API, `text-generation` → generation API)
3. Tries models in configured order with retry (3 attempts, exponential backoff)
4. First successful model is used for that agent
5. If all models fail → deterministic fallback (no crash)

**Supported models (default):**
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `codellama/CodeLlama-7b-Instruct-hf`
- `bigcode/starcoder2-15b`

## Failure Handling

- **No API key?** → Warning logged, LLM layer disabled, deterministic-only. No crash.
- **Model returns bad JSON?** → Try next model. If all fail → deterministic fallback.
- **Hallucination detected?** → LLM weight reduced automatically (default 50% reduction).
- **Timeout / sandbox violation?** → Captured in artifact, deterministic score reflects it.

## Scoring Logic

```
Default:  Final = 60% deterministic + 40% LLM (consensus)
Hallucination flags:  LLM weight reduced (e.g. 40% → 20%)
LLM unavailable:  Final = 100% deterministic
```

Five dimensions (0–10 each):
- **Correctness** (35%) — test pass rate
- **Edge Cases** (20%) — partial credit curve
- **Complexity** (15%) — AST loop nesting analysis
- **Style** (15%) — ruff warnings + heuristics
- **Clarity** (15%) — readability heuristics

## Usage

```bash
# Evaluate all problems and submissions
python cli.py --evaluate all

# Evaluate a specific problem
python cli.py --evaluate problem_1

# Evaluate a specific submission
python cli.py --evaluate problem_1 --submission correct_optimal

# Deterministic-only mode (no LLM)
python cli.py --evaluate all --no-llm

# Verbose logging
python cli.py --evaluate all --verbose
```

## Setup

```bash
pip install -r requirements.txt
```

Set your HuggingFace API key (optional — system works without it):
```bash
export HF_API_KEY=hf_your_token_here
```

## Hardware Requirements

- **Deterministic only (`--no-llm`):** Any machine, no GPU needed
- **With LLM:** Requires internet access for HuggingFace Inference API (no local GPU needed)

## How to Disable LLM

Three ways:
1. CLI flag: `--no-llm`
2. Config: set `llm.enabled: false` in `config.yaml`
3. Environment: don't set `HF_API_KEY` (warning logged, graceful fallback)

## Output

Reports generated in `reports/`:
- `evaluation_report.md` — human-readable summary with per-agent details
- `results.json` — machine-readable full results

## Project Structure

```
Code-Evaluator-V2/
├── cli.py                          # CLI entry point
├── config.yaml                     # Pipeline configuration
├── requirements.txt                # Python dependencies
├── README.md
├── core/
│   ├── orchestrator.py             # Pipeline coordinator
│   ├── deterministic_engine.py     # Sandbox + test runner + rubric scoring
│   └── schema.py                   # Data models
├── agents/
│   ├── test_designer.py            # Test Designer Agent
│   ├── code_reviewer.py            # Code Reviewer Agent
│   ├── complexity_analyst.py       # Complexity Analyst Agent
│   └── consensus_agent.py          # Consensus Agent
├── llm/
│   ├── provider_adapter.py         # Capability-aware HF API adapter
│   ├── model_router.py             # Multi-model router with fallback
│   └── prompt_templates.py         # Agent prompt templates
├── safety/
│   └── hallucination_auditor.py    # Hallucination detection
└── reports/
    ├── evaluation_report.md        # Generated report
    └── results.json                # Generated results
```
