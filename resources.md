# Resources Catalog

This document catalogs all resources gathered for the research project: **Can LLMs Expose What Science Refuses to See?**

---

## Summary

| Resource Type | Count | Location |
|---------------|-------|----------|
| Papers | 20 | papers/ |
| Datasets | 5 | datasets/ |
| Code Repositories | 5 | code/ |

---

## Papers

Total papers downloaded: **20**

| arXiv ID | Title | Year | File | Key Topic |
|----------|-------|------|------|-----------|
| 2505.13259 | From Automation to Autonomy: A Survey on LLMs in Scientific Discovery | 2025 | papers/2505.13259_llms_scientific_discovery_survey.pdf | LLM taxonomy for science |
| 2501.04306 | LLM4SR: Survey on LLMs for Scientific Research | 2025 | papers/2501.04306_llm4sr_survey_llms_scientific_research.pdf | Research gap identification |
| 2503.21248 | ResearchBench: Benchmarking LLMs in Scientific Discovery | 2025 | papers/2503.21248_researchbench.pdf | Hypothesis discovery benchmark |
| 2409.04109 | Can LLMs Generate Novel Research Ideas? | 2024 | papers/2409.04109_can_llms_generate_novel_research_ideas.pdf | Human-LLM comparison study |
| 2410.05080 | ScienceAgentBench | 2024 | papers/2410.05080_scienceagentbench.pdf | Data-driven discovery benchmark |
| 2407.01725 | DiscoveryBench | 2024 | papers/2407.01725_discoverybench.pdf | Discovery tasks benchmark |
| 2411.02429 | IdeaBench | 2024 | papers/2411.02429_ideabench.pdf | Idea generation benchmark |
| 2305.14259 | SciMON: Scientific Inspiration Machines Optimized for Novelty | 2023 | papers/2305.14259_scimon_novelty.pdf | Novelty optimization |
| 2410.13185 | Chain of Ideas: Novel Idea Development with LLM Agents | 2024 | papers/2410.13185_chain_of_ideas.pdf | Idea generation framework |
| 2406.03921 | Knowledge Transfer, Knowledge Gaps, and Knowledge Silos | 2024 | papers/2406.03921_knowledge_gaps_silos_citations.pdf | Citation network analysis |
| 2510.25055 | GAPMAP: Mapping Scientific Knowledge Gaps | 2025 | papers/2510.25055_gapmap_knowledge_gaps.pdf | Knowledge gap detection |
| 2512.05714 | Big Tech-Funded AI Papers: Citation Impact and Bias | 2024 | papers/2512.05714_big_tech_ai_citations.pdf | Funding bias analysis |
| 2410.09510 | Scito2M: 30-Year Cross-disciplinary Scientometric Dataset | 2024 | papers/2410.09510_scito2m_scientometric_analysis.pdf | Scientometric dataset |
| 2408.06292 | The AI Scientist | 2024 | papers/2408.06292_ai_scientist.pdf | Automated scientific discovery |
| 2404.07738 | ResearchAgent | 2024 | papers/2404.07738_researchagent.pdf | LLM research agent |
| 2411.02382 | Knowledge Graph + CoI for Hypothesis Generation | 2024 | papers/2411.02382_kg_coi_hypothesis_generation.pdf | KG-LLM integration |
| 2409.04600 | LLM Automated Systematic Review | 2024 | papers/2409.04600_llm_automated_systematic_review.pdf | LLM for literature review |
| 2412.03531 | Scientific Knowledge Extraction with LLMs | 2024 | papers/2412.03531_scientific_knowledge_extraction_llms.pdf | Knowledge extraction |
| 2504.08619 | Analyzing Publication Types in LLM Papers | 2024 | papers/2504.08619_analyzing_llm_papers.pdf | Bibliometric analysis |
| 2510.10336 | FIND: NSF Awards and Research Outputs Database | 2025 | papers/2510.10336_find_nsf_awards_database.pdf | Funding-publication links |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets available: **5**

| Name | Source | Size | Task | Location | Download Method |
|------|--------|------|------|----------|-----------------|
| DiscoveryBench | HuggingFace | 264 tasks (real) + 903 (synthetic) | Data-driven discovery | datasets/discoverybench/ | `load_dataset("allenai/discoverybench")` |
| ScienceAgentBench | HuggingFace | 102 tasks | Scientific code generation | datasets/scienceagentbench/ | `load_dataset("osunlp/ScienceAgentBench")` |
| NSF Awards | HuggingFace | 523,369 awards | Funding analysis | datasets/nsf_awards/ | `load_dataset("ccm/nsf-awards")` |
| ML-ArXiv-Papers | HuggingFace | 117,592 papers | Paper analysis | datasets/arxiv_papers/ | `load_dataset("CShorten/ML-ArXiv-Papers")` |
| SciCite | HuggingFace | ~8,000 citations | Citation intent | datasets/scicite/ | `load_dataset("allenai/scicite")` |

See `datasets/README.md` for detailed descriptions and download instructions.

### Additional Recommended Datasets (Not Downloaded)

| Name | Source | Size | Notes |
|------|--------|------|-------|
| Semantic Scholar Open Research Corpus (S2ORC) | Semantic Scholar API | 225M+ papers | Requires API key for bulk download |
| SciEvo/Scito2M | ArXiv-based | 2M papers, 30 years | Referenced in paper 2410.09510 |
| GAPMAP | GitHub | 212 paragraphs | https://github.com/UCDenver-ccp/GAPMAP |

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| DiscoveryBench | https://github.com/allenai/discoverybench | Discovery benchmark + agents | code/discoverybench/ |
| ScienceAgentBench | https://github.com/osu-nlp-group/scienceagentbench | Science agent evaluation | code/scienceagentbench/ |
| AI-Scientist | https://github.com/SakanaAI/AI-Scientist | Automated scientific discovery | code/ai-scientist/ |
| Awesome-LLM-Scientific-Discovery | https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery | Paper list + resources | code/awesome-llm-scientific-discovery/ |
| Impact-Big-Tech-Funding | https://github.com/Peerzival/impact-big-tech-funding | Funding bias analysis | code/impact-big-tech-funding/ |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**: Combined web search with arXiv, Semantic Scholar, and Papers with Code
2. **Keyword Strategy**:
   - Primary: "LLM scientific discovery", "research gap identification", "knowledge gaps"
   - Secondary: "bibliometrics AI", "funding bias research", "under-researched topics"
   - Tertiary: "hypothesis generation", "idea generation benchmark"
3. **Dataset Search**: HuggingFace datasets, Semantic Scholar API, GitHub repositories
4. **Repository Search**: GitHub search for paper implementations, official code releases

### Selection Criteria

Papers selected based on:
- Direct relevance to LLMs for scientific discovery
- Focus on research gap/knowledge gap identification
- Funding bias and attention disparity analysis
- Benchmark datasets for evaluation
- Recency (preference for 2024-2025)

Datasets selected based on:
- Compatibility with research hypothesis
- Availability and accessibility
- Scale appropriate for experimentation
- Well-documented structure

### Challenges Encountered

1. **SciCite dataset**: Uses deprecated loading script format; samples not extractable via standard method
2. **GAPMAP dataset**: Not on HuggingFace; requires GitHub clone for access
3. **S2ORC**: Requires Semantic Scholar partner API key for bulk access
4. **Large PDF files**: Some papers (AI Scientist, Scito2M) are very large (11-21MB)

### Gaps and Workarounds

1. **Real-world impact data**: No single dataset links research topics to societal impact metrics. Recommend combining:
   - WHO disease burden data
   - Economic impact databases
   - News/media attention metrics

2. **Cross-domain gap identification**: Limited datasets span multiple scientific domains. DiscoveryBench covers 6 domains; may need to combine with ArXiv full dataset.

3. **Funding-topic mapping**: NSF Awards dataset provides funding data but needs enrichment with topic classification.

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset(s)
- **DiscoveryBench** for structured discovery task evaluation
- **NSF Awards** for funding pattern analysis
- **ArXiv papers** for topic trend analysis

### 2. Baseline Methods
- Direct LLM prompting (GPT-4, Claude)
- RAG-enhanced LLM with Semantic Scholar
- Citation network analysis from Scito2M data

### 3. Evaluation Metrics
- Gap identification precision/recall
- Novelty score (vs. existing literature)
- Cross-domain coverage
- Alignment with external impact indicators

### 4. Code to Adapt/Reuse
- `code/discoverybench/agents/` - Discovery agent implementation
- `code/ai-scientist/` - End-to-end research agent pipeline
- `code/impact-big-tech-funding/` - Funding analysis scripts

---

## API Resources

### Semantic Scholar API
- **Documentation**: https://www.semanticscholar.org/product/api
- **Rate Limits**: 1000 req/sec (unauthenticated), 1 req/sec (authenticated)
- **Datasets API**: https://api.semanticscholar.org/api-docs/datasets
- **Python Client**: `pip install semanticscholar`

### ArXiv API
- **Documentation**: https://arxiv.org/help/api
- **Bulk Access**: https://arxiv.org/help/bulk_data

### NSF Award Search API
- **Documentation**: https://www.nsf.gov/awardsearch/
- **Data Download**: https://www.nsf.gov/awardsearch/download.jsp

---

## File Organization

```
workspace/
├── papers/                      # 20 PDF papers
│   ├── README.md               # Paper descriptions
│   └── *.pdf                   # Paper files
├── datasets/                    # Dataset samples and instructions
│   ├── .gitignore              # Excludes large data files
│   ├── README.md               # Download instructions
│   ├── discoverybench/         # DiscoveryBench samples
│   ├── scienceagentbench/      # ScienceAgentBench samples
│   ├── nsf_awards/             # NSF Awards samples
│   ├── arxiv_papers/           # ArXiv samples
│   └── scicite/                # SciCite samples
├── code/                        # 5 cloned repositories
│   ├── README.md               # Repository descriptions
│   ├── discoverybench/         # Allen AI discovery benchmark
│   ├── scienceagentbench/      # OSU NLP science agent
│   ├── ai-scientist/           # Sakana AI scientist
│   ├── awesome-llm-scientific-discovery/  # Paper list
│   └── impact-big-tech-funding/ # Funding analysis
├── literature_review.md         # Comprehensive literature synthesis
├── resources.md                 # This file
└── .resource_finder_complete    # Completion marker
```
