# Can LLMs Expose What Science Refuses to See?

An experimental study testing whether Large Language Models can identify under-researched but high-impact scientific topics.

## Key Findings

- **LLMs accurately estimate research attention**: GPT-4o's attention scores correlated strongly with expected levels (Spearman r = 0.76, p < 0.001)
- **Gap detection works**: LLMs identify genuine research gaps with large effect size (Cohen's d = 1.18, p = 0.005)
- **Cross-model consistency**: GPT-4o and Claude-3.5-Sonnet show 97% correlation in gap scores
- **Practical capability**: LLMs can spontaneously identify known under-researched areas (antimicrobial resistance, chronic pain, climate adaptation)

## Top Identified Research Gaps

1. **Neglected Tropical Diseases Detection** - Affects 1B+ people, receives low AI attention
2. **AI for Low-Resource Languages** - 7000+ languages, minimal NLP resources
3. **Small-Scale Farmer Decision Support** - Critical for food security in developing regions
4. **Maternal Mortality Risk Prediction** - High deaths in low-resource settings
5. **Educational AI for Underserved Communities** - Digital divide in education

## Project Structure

```
├── REPORT.md                 # Full research report with all findings
├── planning.md               # Experimental design document
├── src/
│   ├── experiment.py         # Main gap detection experiment
│   └── model_comparison.py   # Cross-model validation
├── results/
│   ├── experiment_results.json
│   ├── model_comparison.json
│   └── topic_scores.csv
├── figures/
│   ├── attention_vs_impact.png
│   ├── gap_scores.png
│   ├── validation_boxplots.png
│   └── domain_analysis.png
├── literature_review.md      # Background literature synthesis
└── resources.md              # Gathered datasets and papers
```

## Quick Start

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv add datasets numpy pandas matplotlib seaborn openai httpx scipy scikit-learn tqdm

# Run experiments (requires OPENROUTER_API_KEY env var)
export OPENROUTER_API_KEY="your-key"
python src/experiment.py
python src/model_comparison.py
```

## Methodology

1. **Topic Set**: 20 research topics across medicine, AI/ML, agriculture, education, computing
2. **LLM Scoring**: GPT-4o rates each topic on attention (1-10) and impact (1-10)
3. **Gap Calculation**: gap_score = impact - attention
4. **Validation**: Compare against expected categories from literature
5. **Cross-Validation**: Test consistency with Claude-3.5-Sonnet

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Attention correlation | r = 0.76 | LLMs accurately estimate research activity |
| Gap identification | d = 1.18 | Large effect size distinguishing true gaps |
| Cross-model agreement | r = 0.97 | Consistent across different LLMs |

## Implications

- **Research Policy**: LLMs could audit funding portfolios for systematic gaps
- **Scientific Discovery**: Direct attention to high-impact but neglected problems
- **AI Systems**: Integrate gap detection into automated research agents

## Citation

If you use this work, please cite:

```
Research: Can LLMs Expose What Science Refuses to See?
A study of LLM capabilities in identifying research attention-impact disparities.
December 2025
```

## License

This research code is provided for academic use. See individual dataset licenses in `resources.md`.
