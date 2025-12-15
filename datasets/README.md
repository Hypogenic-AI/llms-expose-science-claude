# Downloaded Datasets

This directory contains datasets for the research project: **Can LLMs Expose What Science Refuses to See?**

Data files are NOT committed to git due to size. Follow the download instructions below.

---

## Dataset 1: DiscoveryBench

### Overview
- **Source**: [allenai/discoverybench](https://huggingface.co/datasets/allenai/discoverybench)
- **Size**: 264 real tasks + 903 synthetic tasks
- **Format**: HuggingFace Dataset
- **Task**: Data-driven scientific discovery
- **Splits**: train (25), test (239)
- **License**: ODC-BY (data), Apache 2.0 (code)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("allenai/discoverybench")
dataset.save_to_disk("datasets/discoverybench/full")
```

**Alternative (GitHub clone):**
```bash
git clone https://github.com/allenai/discoverybench.git
# Data in discoverybench/discoverybench/ folder
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/discoverybench/full")
# Or load directly
from datasets import load_dataset
dataset = load_dataset("allenai/discoverybench")
```

### Sample Data
See `discoverybench/samples_train.json` and `discoverybench/samples_test.json`

### Notes
- Tasks require both statistical analysis and semantic reasoning
- Each task has goal, dataset(s), and metadata
- Gold hypotheses in answer_keys for evaluation

---

## Dataset 2: ScienceAgentBench

### Overview
- **Source**: [osunlp/ScienceAgentBench](https://huggingface.co/datasets/osunlp/ScienceAgentBench)
- **Size**: 102 tasks from 44 peer-reviewed publications
- **Format**: HuggingFace Dataset (metadata) + zip (full data)
- **Task**: Data-driven scientific discovery code generation
- **Disciplines**: Bioinformatics, Computational Chemistry, GIS, Psychology/Cognitive Neuroscience
- **License**: CC-BY-4.0

### Download Instructions

**Metadata only (HuggingFace):**
```python
from datasets import load_dataset
dataset = load_dataset("osunlp/ScienceAgentBench")
# Only validation split available: 102 examples
```

**Full benchmark with data:**
```bash
# Download from GitHub releases
# Password for unzip: scienceagentbench
# Do not redistribute unzipped files
```

### Loading the Dataset
```python
from datasets import load_dataset
dataset = load_dataset("osunlp/ScienceAgentBench")
print(dataset['validation'][0])
```

### Sample Data
See `scienceagentbench/samples.json`

### Notes
- Tasks require generating standalone Python programs
- Expert-validated scientific plausibility
- Part of OpenHands evaluation harness

---

## Dataset 3: NSF Awards

### Overview
- **Source**: [ccm/nsf-awards](https://huggingface.co/datasets/ccm/nsf-awards)
- **Size**: 523,369 awards
- **Format**: HuggingFace Dataset
- **Task**: Funding pattern analysis
- **Coverage**: NSF grants since 1989
- **License**: Public domain (NSF data)

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("ccm/nsf-awards")
# Single train split with 523,369 examples
dataset.save_to_disk("datasets/nsf_awards/full")
```

**Alternative (NSF direct download):**
```bash
# Download by year from:
# https://www.nsf.gov/awardsearch/download.jsp
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/nsf_awards/full")
# Or load directly
from datasets import load_dataset
dataset = load_dataset("ccm/nsf-awards")
```

### Sample Data
See `nsf_awards/samples.json`

### Notes
- Contains award title, abstract, amount, dates, PI info
- Can be linked to publications via acknowledgement sections
- Useful for analyzing funding-research topic correlations

---

## Dataset 4: ML-ArXiv-Papers

### Overview
- **Source**: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
- **Size**: 117,592 papers
- **Format**: HuggingFace Dataset
- **Task**: Paper analysis, topic modeling
- **Coverage**: Machine learning papers from arXiv

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
# Full dataset
dataset = load_dataset("CShorten/ML-ArXiv-Papers")
# Or sample for testing
dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:1000]")
dataset.save_to_disk("datasets/arxiv_papers/sample")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/arxiv_papers/sample")
```

### Sample Data
See `arxiv_papers/samples.json`

### Notes
- Contains title, abstract, authors
- Good for testing topic modeling approaches
- For full arXiv coverage, use arxiv-community/arxiv_dataset

---

## Dataset 5: SciCite

### Overview
- **Source**: [allenai/scicite](https://huggingface.co/datasets/allenai/scicite)
- **Size**: ~8,000 citation contexts
- **Format**: JSON
- **Task**: Citation intent classification
- **Labels**: Method, Background, Result
- **License**: Apache 2.0

### Download Instructions

**Note**: This dataset uses deprecated loading script. Use alternative method:

```python
# Alternative: download from source
import requests
url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
# Or use local files from GitHub
```

### Sample Data
See `scicite/samples.json` (if available)

### Notes
- Useful for understanding citation context
- Can help analyze how different research areas cite each other

---

## Additional Recommended Datasets (Not Downloaded)

### Semantic Scholar Open Research Corpus (S2ORC)
- **Source**: Semantic Scholar API
- **Size**: 225M+ papers
- **Access**: Requires partner API key for bulk download
- **Documentation**: https://www.semanticscholar.org/product/api

**To access:**
```python
# Install client
pip install semanticscholar

# Use API
from semanticscholar import SemanticScholar
sch = SemanticScholar()
paper = sch.get_paper("10.1145/3442188.3445922")
```

### SciEvo/Scito2M
- **Source**: arXiv:2410.09510
- **Size**: 2M+ papers, 30 years of data
- **Contains**: Full text, citations, temporal metadata
- **Status**: Check paper for data availability

### GAPMAP Biomedical Gap Dataset
- **Source**: https://github.com/UCDenver-ccp/GAPMAP
- **Size**: 212 paragraphs from 137 PubMed articles
- **Task**: Knowledge gap identification
- **Contains**: Annotated explicit and implicit gaps

**To access:**
```bash
git clone https://github.com/UCDenver-ccp/GAPMAP.git datasets/gapmap
```

---

## Quick Start

```python
from datasets import load_dataset

# Load primary datasets for the research
discoverybench = load_dataset("allenai/discoverybench")
nsf_awards = load_dataset("ccm/nsf-awards")
scienceagent = load_dataset("osunlp/ScienceAgentBench")

print(f"DiscoveryBench: {len(discoverybench['test'])} test tasks")
print(f"NSF Awards: {len(nsf_awards['train'])} awards")
print(f"ScienceAgentBench: {len(scienceagent['validation'])} tasks")
```

---

## Data Usage Notes

1. **Large Datasets**: NSF Awards and ArXiv datasets are large; consider sampling for initial experiments
2. **Rate Limits**: Semantic Scholar API has rate limits; use bulk downloads for large-scale analysis
3. **Citation**: Cite original papers when using these datasets in publications
4. **License Compliance**: Check individual dataset licenses before commercial use
