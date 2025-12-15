# Literature Review: Can LLMs Expose What Science Refuses to See?

## Research Area Overview

This literature review examines the intersection of Large Language Models (LLMs) and scientific discovery, with a particular focus on their potential to identify under-researched topics and knowledge gaps in scientific literature. The research hypothesis posits that LLMs can systematically identify and expose important scientific problems that are under-researched or ignored by analyzing disparities between well-funded, benchmarked topics and those with high real-world impact but low attention.

The field has seen explosive growth, with LLMs evolving from task-specific automation tools into increasingly autonomous agents that can potentially redefine research processes. This review synthesizes 20 papers covering: (1) LLM-based scientific discovery frameworks, (2) benchmarks for evaluating research ideation capabilities, (3) methods for identifying knowledge gaps and silos, and (4) analysis of funding biases in AI research.

---

## Key Papers

### 1. From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery
- **Authors**: Zheng, Deng, Tsang, Wang, Bai, Wang, Song (HKUST)
- **Year**: 2025
- **Source**: arXiv:2505.13259
- **Key Contribution**: Introduces a foundational three-level taxonomy for LLM involvement in scientific discovery: (i) LLM as Tool, (ii) LLM as Analyst, and (iii) LLM as Scientist. Maps LLM applications across the six stages of the scientific method.
- **Methodology**: Systematic literature review with focus on autonomy levels and evolving LLM roles.
- **Relevance to Our Research**: Provides conceptual framework for understanding how LLMs could be deployed to identify research gaps. The "LLM as Scientist" level represents systems capable of autonomous hypothesis generation and research gap identification.
- **Code Available**: Yes - https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery

### 2. LLM4SR: A Survey on Large Language Models for Scientific Research
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2501.04306
- **Key Contribution**: Comprehensive survey covering LLMs across the entire scientific research cycle. Notes that "MLR-Copilot finds research directions by analyzing the research gaps from input papers."
- **Methodology**: Survey of 260+ LLMs in scientific discovery across various disciplines.
- **Relevance to Our Research**: Directly addresses research gap identification as a core LLM capability. Provides broader perspective on LLM applications beyond AI-focused benchmarks.
- **Code Available**: No

### 3. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers
- **Authors**: Chenglei Si, Diyi Yang, Tatsunori Hashimoto (Stanford)
- **Year**: 2024
- **Source**: arXiv:2409.04109
- **Key Contribution**: First statistically significant comparison between expert NLP researchers and LLM ideation agents. Found that AI-generated ideas scored higher in novelty (5.64/10) vs human ideas (4.84/10), p<0.05.
- **Methodology**: N=104 total participants (49 idea writers, 79 reviewers). Blind review of both LLM and human ideas.
- **Datasets Used**: Custom dataset of NLP research topics
- **Key Findings**: LLMs excel at novelty but struggle with feasibility. LLMs show limitations in self-evaluation and diversity of generation.
- **Relevance to Our Research**: Provides empirical evidence that LLMs can generate novel ideas, supporting the hypothesis that they could identify under-explored research areas.
- **Code Available**: Yes - https://github.com/NoviScl/AI-Researcher

### 4. GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using Large Language Models
- **Authors**: Salem, White, Bada, Hunter (UC Denver, U Chicago)
- **Year**: 2025
- **Source**: arXiv:2510.25055
- **Key Contribution**: Introduces framework for LLM-based identification of both explicit and implicit knowledge gaps in biomedical literature.
- **Methodology**: Two experiments on ~1500 documents across four datasets. Introduces TABI (Toulmin-Abductive Bucketed Inference) for structured reasoning.
- **Datasets Used**: 212 biomedical paragraphs from 137 PubMed articles, manually annotated
- **Key Findings**: LLMs show robust capability in identifying both explicit and implicit knowledge gaps. Larger models often perform better.
- **Relevance to Our Research**: **Directly relevant** - demonstrates that LLMs can systematically identify knowledge gaps, which is central to our hypothesis.
- **Code Available**: Yes - https://github.com/UCDenver-ccp/GAPMAP

### 5. Knowledge Transfer, Knowledge Gaps, and Knowledge Silos in Citation Networks
- **Authors**: Cunningham, Greene (UCD, Insight Centre)
- **Year**: 2024
- **Source**: arXiv:2406.03921
- **Key Contribution**: Novel network analysis framework to study knowledge transfer dynamics from citation data. Identifies "knowledge silos" (isolated application domains) and "knowledge gaps" (opportunities for cross-pollination).
- **Methodology**: Dynamic community detection on cumulative, time-evolving citation networks. Case study on XAI research.
- **Key Findings**: Knowledge transfer between foundational and contemporary topics is limited; certain domains exist as isolated silos; significant gaps exist between related research areas.
- **Relevance to Our Research**: **Highly relevant** - provides methodology for identifying under-researched connections in citation networks that LLMs could leverage.
- **Code Available**: Not mentioned

### 6. Big Tech-Funded AI Papers Have Higher Citation Impact, Greater Insularity, and Larger Recency Bias
- **Authors**: Gnewuch, Wahle, Ruas, Gipp (U Goettingen)
- **Year**: 2024
- **Source**: arXiv:2512.05714
- **Key Contribution**: Quantifies funding bias in AI research by analyzing ~49.8K papers, ~1.8M citations at 10 top AI conferences (1998-2022).
- **Methodology**: Scientometric analysis of industry funding acknowledgements and citation patterns.
- **Key Findings**: Industry-funded work is increasingly insular (citing predominantly other industry-funded papers). Non-funded work may focus more on foundational contributions that are "neglected" by funded research.
- **Datasets Used**: Custom dataset of 49.8K AI papers with funding metadata
- **Relevance to Our Research**: **Critical** - demonstrates that funding shapes research attention, potentially causing neglect of important topics. Provides empirical basis for hypothesis that LLMs could expose such disparities.
- **Code Available**: Yes - https://github.com/Peerzival/impact-big-tech-funding

### 7. SciMON: Scientific Inspiration Machines Optimized for Novelty
- **Authors**: Wang, Downey, Ji, Hope (UIUC, AI2, Hebrew U)
- **Year**: 2023
- **Source**: arXiv:2305.14259
- **Key Contribution**: Framework for generating novel scientific directions grounded in literature. Uses retrieval of "inspirations" from past papers and iteratively optimizes for novelty.
- **Methodology**: Models take background contexts (problems, settings, goals) and output natural language ideas grounded in literature.
- **Key Findings**: GPT-4 tends to generate ideas with low technical depth and novelty; SciMON partially mitigates this.
- **Relevance to Our Research**: Provides methodology for grounding LLM idea generation in literature, which could be adapted for gap identification.
- **Code Available**: Referenced but link not provided in paper

### 8. ResearchAgent: Iterative Research Idea Generation over Scientific Literature
- **Authors**: Baek, Jauhar, Cucerzan, Hwang (KAIST, Microsoft Research)
- **Year**: 2024
- **Source**: arXiv:2404.07738
- **Key Contribution**: System that automatically defines novel problems, proposes methods, and designs experiments using academic knowledge graphs and LLMs.
- **Methodology**: Augments LLM with academic graph and entity-centric knowledge store. Uses multiple LLM-based Reviewing Agents for iterative refinement.
- **Key Findings**: Combining academic graphs with LLMs improves cross-pollination of ideas across domains.
- **Relevance to Our Research**: Demonstrates how LLMs can leverage structured scientific knowledge to identify novel research directions.
- **Code Available**: Not mentioned

### 9. Chain of Ideas: Revolutionizing Research in Novel Idea Development with LLM Agents
- **Authors**: Li et al. (Alibaba DAMO, Zhejiang U)
- **Year**: 2024
- **Source**: arXiv:2410.13185
- **Key Contribution**: CoI agent organizes relevant literature in a chain structure to mirror progressive development in a research domain, facilitating LLMs to capture current advancements and enhance ideation.
- **Methodology**: Literature organization + LLM-based idea generation + Idea Arena evaluation protocol
- **Key Findings**: CoI agent shows comparable quality to humans in research idea generation at minimum cost of $0.50 per idea.
- **Relevance to Our Research**: Provides efficient methodology for LLM-based ideation that could scale to systematic gap analysis.
- **Code Available**: Not mentioned

### 10. DiscoveryBench: Towards Data-Driven Discovery with Large Language Models
- **Authors**: Majumder, Surana, Agarwal, et al. (Allen AI)
- **Year**: 2024
- **Source**: arXiv:2407.01725
- **Key Contribution**: First comprehensive benchmark for data-driven discovery with LLMs. 264 real tasks + 903 synthetic tasks across 6 domains.
- **Methodology**: Tasks extracted from published papers requiring both statistical analysis and semantic reasoning.
- **Key Findings**: Best systems score only 25%, highlighting challenges in autonomous discovery.
- **Datasets Used**: Custom benchmark derived from 20+ published papers
- **Relevance to Our Research**: Provides benchmark for evaluating LLM capabilities in scientific discovery tasks.
- **Code Available**: Yes - https://github.com/allenai/discoverybench
- **Dataset**: https://huggingface.co/datasets/allenai/discoverybench

### 11. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
- **Authors**: Chen et al. (OSU NLP Group)
- **Year**: 2024
- **Source**: arXiv:2410.05080
- **Key Contribution**: 102 tasks from 44 peer-reviewed publications in 4 disciplines (Bioinformatics, Computational Chemistry, GIS, Psychology/Cognitive Neuroscience).
- **Methodology**: Expert-validated tasks with unified Python program output format.
- **Key Findings**: Best agent solves only 32.4% of tasks independently (42.2% with o1-preview), highlighting current limitations.
- **Relevance to Our Research**: Provides rigorous assessment framework that could be adapted for gap identification tasks.
- **Code Available**: Yes - https://github.com/osu-nlp-group/scienceagentbench
- **Dataset**: https://huggingface.co/datasets/osunlp/ScienceAgentBench

### 12. ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2503.21248
- **Key Contribution**: First large-scale benchmark for evaluating LLMs on scientific hypothesis discovery with sub-tasks: inspiration retrieval, hypothesis composition, and hypothesis ranking.
- **Methodology**: Inspiration-based task decomposition
- **Relevance to Our Research**: Provides framework for evaluating hypothesis generation which could be extended to gap identification.
- **Code Available**: Referenced

### 13. IdeaBench: Benchmarking Large Language Models for Research Idea Generation
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2411.02429
- **Key Contribution**: Benchmark for idea generation with reference-based metrics that align with human judgment.
- **Key Findings**: LLMs excel at generating novel ideas but struggle with feasibility.
- **Relevance to Our Research**: Provides evaluation metrics for idea quality that could apply to gap-filling ideas.
- **Code Available**: Referenced

### 14. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
- **Authors**: Lu, Lu, Lange, Foerster, Clune, Ha (Sakana AI, Oxford, UBC)
- **Year**: 2024
- **Source**: arXiv:2408.06292
- **Key Contribution**: First comprehensive framework for fully automatic scientific discovery. Generates ideas, writes code, runs experiments, writes papers, and runs simulated review.
- **Methodology**: End-to-end pipeline costing less than $15 per paper.
- **Key Findings**: Can produce papers exceeding acceptance thresholds at top ML conferences.
- **Relevance to Our Research**: Demonstrates full autonomy potential; gap identification could be integrated into such systems.
- **Code Available**: Yes - https://github.com/SakanaAI/AI-Scientist

### 15. SciEvo (Scito2M): A 2 Million, 30-Year Cross-disciplinary Dataset for Temporal Scientometric Analysis
- **Authors**: Jin, Xiao, Wang, Wang (Georgia Tech, UCLA, William & Mary)
- **Year**: 2024
- **Source**: arXiv:2410.09510
- **Key Contribution**: 2M+ academic publications with comprehensive content and citation graphs spanning 30 years.
- **Methodology**: Retrieved all CC-licensed arXiv papers (1991-2024), enriched with Semantic Scholar data.
- **Key Findings**: Revealed disparities in epistemic cultures and citation practices (e.g., LLM papers cite work 2.48 years old on average vs. 9.71 years for oral history).
- **Relevance to Our Research**: **Highly relevant** - provides data infrastructure for analyzing research attention disparities over time.
- **Code Available**: Referenced

### 16. A Review on Scientific Knowledge Extraction using Large Language Models in Biomedical Sciences
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2412.03531
- **Key Contribution**: Reviews LLM capabilities for knowledge extraction in biomedical sciences.
- **Key Findings**: Significant challenges remain including hallucinations and contextual understanding. Need for unified benchmarks to standardize evaluations.
- **Relevance to Our Research**: Identifies limitations in LLM knowledge extraction that may affect gap identification accuracy.
- **Code Available**: No

### 17. From Funding to Findings (FIND): An Open Database of NSF Awards and Research Outputs
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2510.10336
- **Key Contribution**: Large-scale open resource linking NSF grant proposals to downstream publications (525,927 unique awards).
- **Methodology**: Systematic matching of NSF grants to Crossref publications.
- **Key Findings**: Only 35% of NSF grants have associated publications in Crossref.
- **Datasets Used**: 525,927 NSF awards, linked publications
- **Relevance to Our Research**: **Critical** - provides data for analyzing funding-research topic correlations and identifying underfunded areas.
- **Code Available**: Referenced

### 18. The emergence of Large Language Models as a tool in literature reviews
- **Authors**: Scherbakov, Hubig, Jansari, Bakumenko, Lenert (MUSC, Clemson)
- **Year**: 2024
- **Source**: arXiv:2409.04600
- **Key Contribution**: Systematic review of LLM usage in creating scientific reviews, using LLM tools.
- **Methodology**: Meta-analysis using LLM-based review process.
- **Relevance to Our Research**: Demonstrates LLM capability in literature synthesis, applicable to gap identification.
- **Code Available**: No

### 19. Hypothesis Generation with Knowledge Graphs and LLMs
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2411.02382
- **Key Contribution**: Combines knowledge graphs with LLMs for generating scientific hypotheses.
- **Methodology**: Knowledge graph construction from scientific literature + LLM reasoning
- **Relevance to Our Research**: Provides methodology for structured hypothesis generation that could identify gaps.
- **Code Available**: Referenced

### 20. Analyzing Publication Types in LLM and Bibliometric Databases
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2504.08619
- **Key Contribution**: Analysis of how different publication types are represented across databases.
- **Relevance to Our Research**: Informs data source selection for comprehensive gap analysis.
- **Code Available**: No

---

## Common Methodologies

### Method 1: LLM-Based Literature Search and Synthesis
Used in: [1, 2, 7, 8, 9, 18]
- Retrieval-augmented generation (RAG) for grounding ideas in literature
- Academic graph integration for cross-domain knowledge
- Iterative refinement through peer review simulation

### Method 2: Benchmark-Based Evaluation
Used in: [3, 10, 11, 12, 13]
- Human expert comparison studies
- Task decomposition for rigorous assessment
- Reference-based metrics aligned with human judgment
- Multi-domain validation

### Method 3: Citation Network Analysis
Used in: [5, 6, 15]
- Dynamic community detection
- Knowledge flow mapping
- Temporal evolution analysis
- Funding acknowledgement extraction

### Method 4: Knowledge Gap Detection
Used in: [4, 5]
- Explicit gap identification via lexical cues
- Implicit gap inference via abductive reasoning
- Toulmin argument model for structured reasoning

---

## Standard Baselines

For idea generation evaluation:
- Direct LLM prompting (GPT-4, Claude, open-weight models)
- RAG-enhanced LLM systems
- Human expert ideas (for comparison)

For gap identification:
- Traditional topic modeling
- Citation count analysis
- Keyword frequency analysis

For scientific discovery benchmarks:
- OpenHands CodeAct framework
- Self-debug approaches
- Direct prompting baselines

---

## Evaluation Metrics

### For Idea/Gap Quality:
- **Novelty**: How different from existing work (human judgment or embedding distance)
- **Feasibility**: Practical implementability
- **Relevance**: Alignment with research goals
- **Impact Potential**: Estimated importance

### For Benchmark Tasks:
- Task completion rate
- Code execution success
- Result accuracy (vs. ground truth)
- Cost per task

### For Gap Identification:
- Precision/recall against expert-annotated gaps
- Coverage of gap types (explicit vs. implicit)
- Cross-domain generalization

---

## Datasets in the Literature

| Dataset | Used By | Task | Size |
|---------|---------|------|------|
| DiscoveryBench | [10] | Data-driven discovery | 264 real + 903 synthetic tasks |
| ScienceAgentBench | [11] | Scientific code generation | 102 tasks, 4 disciplines |
| ResearchBench | [12] | Hypothesis discovery | Large-scale |
| IdeaBench | [13] | Idea generation | Various NLP topics |
| SciEvo/Scito2M | [15] | Scientometric analysis | 2M+ papers, 30 years |
| GAPMAP | [4] | Knowledge gap detection | 1500 documents |
| Big Tech Funding | [6] | Funding analysis | 49.8K papers, 4.1M citations |
| FIND (NSF) | [17] | Funding-publication links | 525K awards |

---

## Gaps and Opportunities

### Gap 1: Limited Focus on Under-Researched Topics
Most existing work focuses on generating ideas within well-established research areas. Little work systematically identifies topics that are under-researched relative to their real-world importance.

### Gap 2: Funding-Attention Disparity Analysis
While [6] shows funding influences research attention, no system currently combines LLM capabilities with funding data to automatically identify high-impact but underfunded research areas.

### Gap 3: Cross-Domain Gap Identification
Current methods largely work within single domains. Systematic cross-domain gap identification (e.g., methods from physics applicable to biology) remains underexplored.

### Gap 4: Real-World Impact Assessment
Existing benchmarks focus on scientific novelty but lack systematic integration of real-world impact metrics (disease burden, economic impact, societal need).

### Gap 5: Temporal Gap Evolution
Limited work tracks how research gaps emerge, persist, or close over time.

---

## Recommendations for Our Experiment

### Recommended Datasets:
1. **Primary**: DiscoveryBench (structured discovery tasks) + SciEvo/Scito2M (30-year citation data)
2. **Secondary**: NSF Awards (ccm/nsf-awards on HuggingFace) for funding analysis
3. **Tertiary**: ArXiv papers dataset for full-text analysis

### Recommended Baselines:
1. Direct GPT-4/Claude prompting for gap identification
2. RAG-enhanced system with Semantic Scholar API
3. Citation network analysis (traditional baseline)
4. Topic modeling (LDA/BERTopic) for comparison

### Recommended Metrics:
1. Gap identification precision/recall (against expert annotations)
2. Novelty of identified gaps (embedding distance from existing work)
3. Alignment with real-world impact indicators
4. Cross-domain coverage

### Methodological Considerations:
1. **Data Contamination**: Use papers from 2024-2025 to reduce training data overlap
2. **Ground Truth**: Create expert-annotated dataset of "known gaps" that were later filled
3. **Validation**: Partner with domain experts to validate identified gaps
4. **Bias Mitigation**: Control for LLM biases toward well-documented topics

---

## Conclusion

The literature reveals a rapidly evolving field where LLMs are progressing from automation tools to increasingly autonomous research agents. Current work has established strong foundations in idea generation, benchmark evaluation, and citation analysis. However, a significant opportunity exists to combine these capabilities for systematically identifying under-researched but high-impact topics.

The hypothesis that LLMs can expose what science "refuses to see" is supported by:
1. Demonstrated LLM capabilities in novelty generation [3, 7, 8, 9]
2. Successful knowledge gap identification in biomedical literature [4]
3. Evidence of funding-driven research attention biases [6]
4. Available large-scale datasets linking funding to publications [17]

The key challenge lies in defining and measuring "high real-world impact but low attention" - this requires integrating LLM capabilities with external data sources on disease burden, economic impact, and societal needs that go beyond academic citation metrics.
