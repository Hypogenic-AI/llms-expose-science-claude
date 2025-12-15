# Cloned Repositories

This directory contains code repositories relevant to the research project: **Can LLMs Expose What Science Refuses to See?**

---

## Repository 1: DiscoveryBench

- **URL**: https://github.com/allenai/discoverybench
- **Purpose**: Benchmark and agents for data-driven scientific discovery
- **Location**: `code/discoverybench/`
- **License**: Apache 2.0 (code), ODC-BY (data)

### Key Files
- `agents/discovery_agent.py` - Discovery agent implementation
- `eval/discovery_eval.py` - Evaluation scripts
- `discoverybench/` - Benchmark data

### How to Use
```bash
cd code/discoverybench
pip install -r requirements.txt

# Run discovery agent
python agents/discovery_agent.py --task <task_file>

# Evaluate results
python eval/discovery_eval.py --predictions <pred_file> --gold <gold_file>
```

### Relevance
- Provides baseline agents for discovery tasks
- Evaluation framework for measuring hypothesis quality
- Can be adapted for gap identification tasks

---

## Repository 2: ScienceAgentBench

- **URL**: https://github.com/osu-nlp-group/scienceagentbench
- **Purpose**: Rigorous assessment of language agents for scientific discovery
- **Location**: `code/scienceagentbench/`
- **License**: MIT

### Key Files
- `benchmark/` - Task definitions and data
- `evaluation/` - Evaluation scripts
- `agents/` - Agent implementations

### How to Use
```bash
cd code/scienceagentbench
pip install -r requirements.txt

# See README for detailed instructions
# Note: Full data requires password-protected zip
```

### Relevance
- Multi-discipline evaluation framework
- Code generation tasks for scientific analysis
- Provides baselines: direct prompting, OpenHands CodeAct, self-debug

---

## Repository 3: AI-Scientist

- **URL**: https://github.com/SakanaAI/AI-Scientist
- **Purpose**: Fully automated open-ended scientific discovery
- **Location**: `code/ai-scientist/`
- **License**: Apache 2.0

### Key Files
- `ai_scientist/` - Main framework
  - `generate_ideas.py` - Idea generation
  - `perform_experiments.py` - Experiment execution
  - `perform_writeup.py` - Paper writing
  - `perform_review.py` - Automated review
- `templates/` - Experiment templates

### How to Use
```bash
cd code/ai-scientist
pip install -r requirements.txt

# Generate ideas
python ai_scientist/generate_ideas.py --model gpt-4

# Full pipeline (requires GPU, API keys)
python launch_scientist.py
```

### Relevance
- Complete end-to-end research pipeline
- Idea generation component can be adapted for gap identification
- Automated review provides evaluation baseline

---

## Repository 4: Awesome-LLM-Scientific-Discovery

- **URL**: https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery
- **Purpose**: Curated list of papers on LLMs in scientific discovery
- **Location**: `code/awesome-llm-scientific-discovery/`
- **License**: MIT

### Key Files
- `README.md` - Comprehensive paper list organized by topic
- Links to papers, code, and datasets

### How to Use
- Reference for finding related work
- Source of additional baselines and datasets
- Categorized by scientific method stages

### Relevance
- Literature reference for the field
- Identifies key papers and trends
- Organized by our target taxonomy (Tool/Analyst/Scientist)

---

## Repository 5: Impact-Big-Tech-Funding

- **URL**: https://github.com/Peerzival/impact-big-tech-funding
- **Purpose**: Analysis of Big Tech funding impact on AI research
- **Location**: `code/impact-big-tech-funding/`
- **License**: Not specified

### Key Files
- `data/` - Processed paper and citation data
- `analysis/` - Jupyter notebooks for analysis
- `scripts/` - Data processing scripts

### How to Use
```bash
cd code/impact-big-tech-funding
pip install -r requirements.txt

# Run analysis notebooks
jupyter notebook analysis/
```

### Relevance
- **Critical** for understanding funding biases
- Contains processed data of ~49.8K papers
- Citation analysis methodology applicable to our research
- Demonstrates how to extract and analyze funding acknowledgements

---

## Quick Setup

Install dependencies for all repositories:

```bash
# DiscoveryBench
pip install datasets transformers

# ScienceAgentBench
pip install openai anthropic

# AI-Scientist
pip install openai anthropic transformers torch

# Impact analysis
pip install pandas numpy jupyter matplotlib seaborn
```

---

## Repository Overview by Use Case

### For Idea/Gap Generation
- `ai-scientist/ai_scientist/generate_ideas.py`
- `discoverybench/agents/discovery_agent.py`

### For Evaluation
- `discoverybench/eval/`
- `scienceagentbench/evaluation/`

### For Citation/Funding Analysis
- `impact-big-tech-funding/analysis/`
- `impact-big-tech-funding/scripts/`

### For Literature Reference
- `awesome-llm-scientific-discovery/README.md`

---

## Integration Notes

1. **API Keys Required**: AI-Scientist and some agents require OpenAI/Anthropic API keys
2. **GPU Recommended**: Full AI-Scientist pipeline benefits from GPU for local models
3. **Data Downloads**: Some repositories require separate data downloads (see individual READMEs)
4. **Python Version**: Most repositories tested with Python 3.9+
