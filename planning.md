# Research Plan: Can LLMs Expose What Science Refuses to See?

## Research Question

Can large language models (LLMs) systematically identify and expose important scientific problems that are under-researched or ignored, by analyzing disparities between well-funded/benchmarked topics and those with high real-world impact but low attention?

## Background and Motivation

Current AI for science accelerates existing research agendas but may perpetuate existing biases in research attention. Well-funded problems with clear benchmarks receive disproportionate attention, while important societal issues (e.g., chronic diseases, under-resourced communities) may be systematically neglected. If LLMs can detect these disparities, they could serve as "gap detectors" to make science more accountable to real-world needs.

**Key insights from literature review:**
1. LLMs demonstrate capability to generate novel research ideas (Si et al., 2024: AI ideas scored higher in novelty than human experts)
2. GAPMAP shows LLMs can identify both explicit and implicit knowledge gaps in biomedical literature
3. Big Tech funding analysis reveals increasing insularity in funded research (Gnewuch et al., 2024)
4. NSF Awards database (525K+ awards) provides funding data linkable to research topics

## Hypothesis Decomposition

We decompose the main hypothesis into three testable sub-hypotheses:

### H1: LLMs can accurately identify research topics with high funding/publication volume
- **Test**: Compare LLM-identified "hot topics" against actual publication counts and funding amounts
- **Success metric**: Correlation > 0.6 between LLM attention scores and actual publication/funding data

### H2: LLMs can identify topics with low research attention relative to stated societal importance
- **Test**: Use LLM to score topics on both "research attention" and "societal impact", then identify gaps
- **Success metric**: Identify at least 5 topics with high impact scores but low attention scores, validated by domain knowledge

### H3: LLM-identified gaps align with known under-researched areas documented in literature
- **Test**: Compare LLM gap identification against expert-documented research gaps (from GAPMAP, systematic reviews)
- **Success metric**: Precision > 0.5 and recall > 0.3 against expert-annotated gaps

## Proposed Methodology

### Approach Overview

We will conduct a multi-stage experiment:

1. **Stage 1: Research Attention Mapping** - Use LLMs to estimate research attention for a set of scientific topics based on publication abstracts
2. **Stage 2: Impact Assessment** - Use LLMs to score the same topics on real-world societal impact
3. **Stage 3: Gap Identification** - Identify topics where Impact >> Attention
4. **Stage 4: Validation** - Validate findings against actual publication counts and expert-identified gaps

### Experimental Steps

#### Step 1: Dataset Preparation
**Rationale**: We need a diverse set of scientific topics with ground-truth publication/funding data.

- Load NSF Awards dataset (sample ~5,000 awards across different directorates)
- Extract research topics using keyword extraction
- Group into ~50-100 topic clusters
- Calculate actual publication attention from arXiv papers dataset

#### Step 2: LLM Research Attention Scoring
**Rationale**: Test whether LLMs can estimate research activity without seeing actual counts.

- Prompt LLM (GPT-4.1 or Claude) with topic descriptions
- Ask: "On a scale of 1-10, how much research attention does this topic receive in academic literature?"
- Compare against actual publication counts

#### Step 3: LLM Societal Impact Scoring
**Rationale**: LLMs have broad knowledge of societal issues and can assess impact.

- Prompt same LLM with same topics
- Ask: "On a scale of 1-10, how important is this topic for societal well-being?"
- Include criteria: health impact, economic impact, number of people affected, urgency

#### Step 4: Gap Identification
**Rationale**: Topics with high impact but low attention are "gaps."

- Calculate gap_score = impact_score - attention_score
- Rank topics by gap_score
- Identify top 20 gaps

#### Step 5: Validation
**Rationale**: Validate that identified gaps are meaningful.

- Compare attention estimates against actual NSF funding amounts
- Compare against ArXiv publication counts by topic
- Qualitative analysis of top gaps against known under-researched areas

### Baselines

1. **Random baseline**: Random attention and impact scores
2. **Keyword frequency baseline**: Use keyword frequency in abstracts as attention proxy
3. **Topic modeling baseline**: Use BERTopic to identify topics, cluster by size

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Attention estimation correlation | Pearson r between LLM scores and actual publication counts | > 0.5 |
| Impact-Attention agreement | Cohen's kappa for gap identification across multiple runs | > 0.6 |
| Gap validity | Expert evaluation of top 10 identified gaps | > 50% rated as valid |
| Cross-model consistency | Agreement between GPT-4 and Claude gap identification | > 70% overlap |

### Statistical Analysis Plan

1. **Correlation analysis**: Pearson/Spearman correlation for continuous metrics
2. **Agreement analysis**: Cohen's kappa for categorical classifications
3. **Significance testing**: Bootstrap confidence intervals (n=1000) for all correlations
4. **Multiple comparisons**: Bonferroni correction for multi-model comparisons
5. **Effect sizes**: Report Cohen's d for all significant differences

## Expected Outcomes

### If hypothesis is supported:
- LLM attention estimates correlate moderately-strongly with actual metrics (r > 0.5)
- LLM identifies gaps that experts agree are under-researched
- Cross-model consistency is high, suggesting robust signal

### If hypothesis is refuted:
- LLM attention estimates show low correlation with reality
- Identified "gaps" are artifacts of LLM biases (e.g., toward popular topics)
- High variance across runs suggests unreliable signal

### Mixed outcomes:
- LLMs may be better at identifying gaps in some domains than others
- Some gap types (explicit vs. implicit) may be easier to detect

## Timeline and Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Setup & Data Prep | 20 min | Environment ready, datasets loaded |
| LLM Experiments | 60 min | All API calls complete, raw results saved |
| Analysis | 30 min | Statistical analysis complete |
| Documentation | 30 min | REPORT.md written |

## Potential Challenges

1. **API rate limits**: Mitigate with batching and caching
2. **LLM inconsistency**: Run multiple trials, average results
3. **Ground truth availability**: Use proxy metrics (funding, publication counts)
4. **Topic granularity**: Topics too broad/narrow may confound results
5. **LLM training bias**: Models may have more knowledge of popular topics

## Success Criteria

The research will be considered successful if:

1. [REQUIRED] Experiments complete with real LLM API calls
2. [REQUIRED] Statistical analysis shows interpretable patterns
3. [DESIRED] At least one sub-hypothesis receives support (p < 0.05)
4. [DESIRED] Identified gaps are qualitatively plausible
5. [BONUS] Cross-model validation shows consistency

## Resources to Use

### Datasets
- NSF Awards (ccm/nsf-awards) - funding patterns
- ML-ArXiv-Papers (CShorten/ML-ArXiv-Papers) - publication attention
- DiscoveryBench (for methodology inspiration)

### APIs
- OpenAI GPT-4.1 (via environment variable)
- OpenRouter (for model comparison)

### Code to Adapt
- code/impact-big-tech-funding/ - funding analysis patterns
- code/discoverybench/agents/ - LLM prompting patterns

## Experimental Configuration

```python
config = {
    'seed': 42,
    'n_topics': 50,
    'n_nsf_samples': 5000,
    'n_arxiv_samples': 10000,
    'llm_trials': 3,
    'models': ['gpt-4o', 'claude-sonnet'],
    'temperature': 0.3,
    'bootstrap_n': 1000
}
```
