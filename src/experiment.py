"""
Experiment: Can LLMs Expose What Science Refuses to See?

This script tests whether LLMs can identify under-researched topics by
analyzing disparities between research attention and societal importance.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
import httpx
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
CONFIG = {
    'seed': SEED,
    'n_topics_per_domain': 10,
    'llm_trials': 2,  # Number of independent LLM evaluations per topic
    'temperature': 0.3,
    'max_tokens': 500,
    'bootstrap_n': 1000
}


def get_openrouter_client() -> OpenAI:
    """Get OpenRouter client for model access."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def get_openai_client() -> OpenAI:
    """Get OpenAI client for model access."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def create_research_topics() -> List[Dict[str, Any]]:
    """
    Create a diverse set of research topics spanning well-funded and under-researched areas.

    Topics are organized into categories:
    - Well-funded/high-attention areas (AI/ML, Cancer, etc.)
    - Under-researched but high-impact areas (as documented in literature)
    - Control topics for calibration
    """
    topics = [
        # === WELL-FUNDED / HIGH-ATTENTION AREAS (from Big Tech funding analysis) ===
        {
            "id": "t1",
            "name": "Large Language Model Pre-training",
            "description": "Training massive neural language models on internet-scale text corpora for general-purpose AI",
            "domain": "AI/ML",
            "expected_attention": "high",
            "expected_impact": "medium"
        },
        {
            "id": "t2",
            "name": "Deep Reinforcement Learning for Games",
            "description": "Using deep learning combined with reinforcement learning to master complex games like Go, Chess, and video games",
            "domain": "AI/ML",
            "expected_attention": "high",
            "expected_impact": "low"
        },
        {
            "id": "t3",
            "name": "Computer Vision for Autonomous Vehicles",
            "description": "Developing perception systems for self-driving cars using cameras, LiDAR, and sensor fusion",
            "domain": "AI/ML",
            "expected_attention": "high",
            "expected_impact": "medium"
        },
        {
            "id": "t4",
            "name": "Breast Cancer Detection with AI",
            "description": "Using machine learning to analyze mammograms and detect early signs of breast cancer",
            "domain": "Medicine",
            "expected_attention": "high",
            "expected_impact": "high"
        },
        {
            "id": "t5",
            "name": "Drug Discovery with Deep Learning",
            "description": "Applying neural networks to predict molecular properties and accelerate pharmaceutical development",
            "domain": "Medicine",
            "expected_attention": "high",
            "expected_impact": "high"
        },

        # === UNDER-RESEARCHED BUT HIGH-IMPACT AREAS (from literature on research gaps) ===
        {
            "id": "t6",
            "name": "Gestational Diabetes Mellitus Management",
            "description": "Developing tools to monitor and manage diabetes that develops during pregnancy, affecting 14% of pregnancies globally",
            "domain": "Medicine",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t7",
            "name": "Endometriosis Diagnosis and Treatment",
            "description": "Research on chronic condition causing tissue similar to uterine lining to grow outside the uterus, affecting 10% of women",
            "domain": "Medicine",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t8",
            "name": "AI for Low-Resource Languages",
            "description": "Developing NLP technologies for the 7000+ languages spoken globally that have minimal digital representation",
            "domain": "AI/ML",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t9",
            "name": "Neglected Tropical Diseases Detection",
            "description": "Using AI to diagnose diseases like Chagas, leishmaniasis, and schistosomiasis affecting 1+ billion people in poverty",
            "domain": "Medicine",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t10",
            "name": "Small-Scale Farmer Decision Support",
            "description": "AI systems to help subsistence farmers in developing regions make planting, irrigation, and harvest decisions",
            "domain": "Agriculture",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t11",
            "name": "Chronic Pain Assessment and Management",
            "description": "Developing objective measures and personalized treatments for chronic pain affecting 20% of adults globally",
            "domain": "Medicine",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t12",
            "name": "Maternal Mortality Risk Prediction",
            "description": "Using AI to predict and prevent maternal deaths, especially in low-resource settings with high maternal mortality",
            "domain": "Medicine",
            "expected_attention": "low",
            "expected_impact": "high"
        },
        {
            "id": "t13",
            "name": "AI for Disability Assistance",
            "description": "Developing AI-powered tools to assist people with physical and cognitive disabilities in daily life",
            "domain": "AI/ML",
            "expected_attention": "low",
            "expected_impact": "high"
        },

        # === CONTROL / CALIBRATION TOPICS ===
        {
            "id": "t14",
            "name": "Quantum Computing Error Correction",
            "description": "Developing algorithms to detect and correct errors in quantum computers to enable practical quantum advantage",
            "domain": "Computing",
            "expected_attention": "medium",
            "expected_impact": "medium"
        },
        {
            "id": "t15",
            "name": "Cryptocurrency Mining Optimization",
            "description": "Improving the efficiency of proof-of-work algorithms for Bitcoin and other cryptocurrency mining",
            "domain": "Computing",
            "expected_attention": "medium",
            "expected_impact": "low"
        },
        {
            "id": "t16",
            "name": "Social Media Recommendation Algorithms",
            "description": "Developing algorithms to personalize content feeds and maximize user engagement on social platforms",
            "domain": "AI/ML",
            "expected_attention": "high",
            "expected_impact": "low"
        },
        {
            "id": "t17",
            "name": "Climate Change Prediction Models",
            "description": "Using machine learning to improve climate predictions and model the impacts of global warming",
            "domain": "Environment",
            "expected_attention": "medium",
            "expected_impact": "high"
        },
        {
            "id": "t18",
            "name": "Mental Health Chatbots",
            "description": "AI-powered conversational agents providing mental health support and crisis intervention",
            "domain": "Medicine",
            "expected_attention": "medium",
            "expected_impact": "high"
        },
        {
            "id": "t19",
            "name": "Rare Disease Diagnosis AI",
            "description": "Using AI to accelerate diagnosis of rare diseases affecting small patient populations",
            "domain": "Medicine",
            "expected_attention": "medium",
            "expected_impact": "medium"
        },
        {
            "id": "t20",
            "name": "Educational AI for Underserved Communities",
            "description": "Developing AI tutoring systems accessible to students in low-income and rural areas globally",
            "domain": "Education",
            "expected_attention": "low",
            "expected_impact": "high"
        }
    ]

    return topics


def query_llm_for_scores(client: OpenAI, topics: List[Dict], model: str = "openai/gpt-4o") -> Dict[str, Dict]:
    """
    Query an LLM to score topics on research attention and societal impact.

    Returns dict mapping topic_id to scores.
    """
    results = {}

    for topic in tqdm(topics, desc=f"Querying {model}"):
        prompt = f"""You are an expert in research trends and societal impact assessment.

For the following research topic, please provide two scores from 1-10:

TOPIC: {topic['name']}
DESCRIPTION: {topic['description']}

SCORES TO PROVIDE:

1. RESEARCH ATTENTION (1-10): How much academic research attention does this topic currently receive?
   - 1 = Almost no academic papers, very obscure topic
   - 5 = Moderate attention, some dedicated research groups
   - 10 = Extremely active field with thousands of papers annually

2. SOCIETAL IMPACT (1-10): How important is this topic for human well-being and societal benefit?
   - 1 = Minimal real-world impact on human lives
   - 5 = Moderate impact, affects specific groups
   - 10 = Critical importance, affects billions of people's health, safety, or quality of life

Respond in exactly this JSON format:
{{"attention_score": <number 1-10>, "impact_score": <number 1-10>, "attention_reasoning": "<1 sentence>", "impact_reasoning": "<1 sentence>"}}

Only output the JSON, nothing else."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=CONFIG['temperature'],
                max_tokens=CONFIG['max_tokens']
            )

            content = response.choices[0].message.content.strip()
            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content)
            results[topic['id']] = {
                'attention_score': float(parsed['attention_score']),
                'impact_score': float(parsed['impact_score']),
                'attention_reasoning': parsed.get('attention_reasoning', ''),
                'impact_reasoning': parsed.get('impact_reasoning', ''),
                'model': model
            }

        except Exception as e:
            print(f"Error processing topic {topic['id']}: {e}")
            results[topic['id']] = {
                'attention_score': None,
                'impact_score': None,
                'error': str(e),
                'model': model
            }

    return results


def query_llm_for_gap_identification(client: OpenAI, model: str = "openai/gpt-4o") -> Dict:
    """
    Ask LLM to directly identify under-researched but important topics.
    Tests the LLM's ability to spontaneously identify research gaps.
    """
    prompt = """You are an expert in scientific research trends and societal needs.

TASK: Identify 5 research topics that are UNDER-RESEARCHED relative to their SOCIETAL IMPORTANCE.

These should be topics where:
1. The real-world impact (health, safety, well-being) is HIGH
2. But academic/industry research attention is disproportionately LOW
3. The gap between impact and attention is notable

For each topic, explain:
- Why the societal impact is high
- Why research attention is low (funding biases, stigma, complexity, etc.)

Respond in exactly this JSON format:
{
  "gaps": [
    {
      "topic": "<topic name>",
      "description": "<brief description>",
      "impact_explanation": "<why high impact>",
      "attention_explanation": "<why low attention>",
      "estimated_impact_score": <1-10>,
      "estimated_attention_score": <1-10>
    }
  ]
}

Only output the JSON, nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Slightly higher for more diverse generation
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return {
            'model': model,
            'gaps': json.loads(content)['gaps'],
            'raw_response': response.choices[0].message.content
        }

    except Exception as e:
        return {'model': model, 'error': str(e)}


def analyze_results(topics: List[Dict], llm_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Combine topics with LLM scores for analysis.
    """
    rows = []
    for topic in topics:
        result = llm_results.get(topic['id'], {})
        rows.append({
            'topic_id': topic['id'],
            'name': topic['name'],
            'domain': topic['domain'],
            'expected_attention': topic['expected_attention'],
            'expected_impact': topic['expected_impact'],
            'llm_attention': result.get('attention_score'),
            'llm_impact': result.get('impact_score'),
            'attention_reasoning': result.get('attention_reasoning', ''),
            'impact_reasoning': result.get('impact_reasoning', '')
        })

    df = pd.DataFrame(rows)

    # Calculate gap score (impact - attention)
    df['gap_score'] = df['llm_impact'] - df['llm_attention']

    # Map expected values to numeric for correlation
    attention_map = {'low': 1, 'medium': 2, 'high': 3}
    impact_map = {'low': 1, 'medium': 2, 'high': 3}

    df['expected_attention_num'] = df['expected_attention'].map(attention_map)
    df['expected_impact_num'] = df['expected_impact'].map(impact_map)

    return df


def compute_validation_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute validation metrics comparing LLM scores to expected values.
    """
    # Filter to rows with valid scores
    valid_df = df.dropna(subset=['llm_attention', 'llm_impact'])

    if len(valid_df) < 5:
        return {'error': 'Insufficient valid data points'}

    # Correlation between LLM attention scores and expected attention
    attention_corr, attention_p = stats.spearmanr(
        valid_df['expected_attention_num'],
        valid_df['llm_attention']
    )

    # Correlation between LLM impact scores and expected impact
    impact_corr, impact_p = stats.spearmanr(
        valid_df['expected_impact_num'],
        valid_df['llm_impact']
    )

    # Gap identification accuracy: do high expected_impact + low expected_attention
    # topics have higher gap scores?
    high_impact_low_attention = valid_df[
        (valid_df['expected_impact'] == 'high') &
        (valid_df['expected_attention'] == 'low')
    ]
    other_topics = valid_df[
        ~((valid_df['expected_impact'] == 'high') &
          (valid_df['expected_attention'] == 'low'))
    ]

    if len(high_impact_low_attention) > 0 and len(other_topics) > 0:
        gap_ttest = stats.ttest_ind(
            high_impact_low_attention['gap_score'].values,
            other_topics['gap_score'].values
        )
        gap_effect_size = (
            high_impact_low_attention['gap_score'].mean() -
            other_topics['gap_score'].mean()
        ) / valid_df['gap_score'].std()
    else:
        gap_ttest = None
        gap_effect_size = None

    return {
        'n_valid': len(valid_df),
        'attention_correlation': {
            'spearman_r': attention_corr,
            'p_value': attention_p
        },
        'impact_correlation': {
            'spearman_r': impact_corr,
            'p_value': impact_p
        },
        'gap_identification': {
            'n_true_gaps': len(high_impact_low_attention),
            'n_other': len(other_topics),
            'mean_gap_score_true': high_impact_low_attention['gap_score'].mean() if len(high_impact_low_attention) > 0 else None,
            'mean_gap_score_other': other_topics['gap_score'].mean() if len(other_topics) > 0 else None,
            't_statistic': gap_ttest.statistic if gap_ttest else None,
            'p_value': gap_ttest.pvalue if gap_ttest else None,
            'effect_size_d': gap_effect_size
        }
    }


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations for the analysis."""

    valid_df = df.dropna(subset=['llm_attention', 'llm_impact'])

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Attention vs Impact scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by expected attention level
    colors = {'low': 'red', 'medium': 'orange', 'high': 'green'}
    for level in ['low', 'medium', 'high']:
        subset = valid_df[valid_df['expected_attention'] == level]
        ax.scatter(subset['llm_attention'], subset['llm_impact'],
                  c=colors[level], label=f'Expected attention: {level}',
                  s=100, alpha=0.7, edgecolors='black')

    # Add topic labels
    for _, row in valid_df.iterrows():
        ax.annotate(row['name'][:20] + '...',
                   (row['llm_attention'], row['llm_impact']),
                   fontsize=7, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')

    # Add diagonal line (attention = impact)
    ax.plot([1, 10], [1, 10], 'k--', alpha=0.3, label='Attention = Impact')

    ax.set_xlabel('LLM-Estimated Research Attention (1-10)', fontsize=12)
    ax.set_ylabel('LLM-Estimated Societal Impact (1-10)', fontsize=12)
    ax.set_title('Research Attention vs Societal Impact\n(Topics above diagonal = under-researched)', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_vs_impact.png', dpi=150)
    plt.close()

    # Figure 2: Gap score by topic
    fig, ax = plt.subplots(figsize=(12, 8))

    sorted_df = valid_df.sort_values('gap_score', ascending=True)

    colors = ['green' if g > 0 else 'red' for g in sorted_df['gap_score']]
    bars = ax.barh(range(len(sorted_df)), sorted_df['gap_score'], color=colors, alpha=0.7)

    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['name'].str[:40], fontsize=9)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Gap Score (Impact - Attention)', fontsize=12)
    ax.set_title('Research Gap Identification\n(Positive = Under-researched, Negative = Over-researched)', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/gap_scores.png', dpi=150)
    plt.close()

    # Figure 3: Box plot by expected attention category
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Attention scores by expected attention
    order = ['low', 'medium', 'high']
    valid_df['expected_attention'] = pd.Categorical(valid_df['expected_attention'], categories=order, ordered=True)

    sns.boxplot(data=valid_df, x='expected_attention', y='llm_attention', ax=axes[0], palette='RdYlGn')
    axes[0].set_xlabel('Expected Attention Level', fontsize=12)
    axes[0].set_ylabel('LLM Attention Score', fontsize=12)
    axes[0].set_title('LLM Attention Scores by Expected Level', fontsize=14)

    # Impact scores by expected impact
    valid_df['expected_impact'] = pd.Categorical(valid_df['expected_impact'], categories=order, ordered=True)

    sns.boxplot(data=valid_df, x='expected_impact', y='llm_impact', ax=axes[1], palette='RdYlGn')
    axes[1].set_xlabel('Expected Impact Level', fontsize=12)
    axes[1].set_ylabel('LLM Impact Score', fontsize=12)
    axes[1].set_title('LLM Impact Scores by Expected Level', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_boxplots.png', dpi=150)
    plt.close()

    # Figure 4: Domain-level analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    domain_stats = valid_df.groupby('domain').agg({
        'llm_attention': 'mean',
        'llm_impact': 'mean',
        'gap_score': 'mean'
    }).reset_index()

    x = range(len(domain_stats))
    width = 0.3

    ax.bar([i - width for i in x], domain_stats['llm_attention'], width, label='Attention', color='blue', alpha=0.7)
    ax.bar(x, domain_stats['llm_impact'], width, label='Impact', color='green', alpha=0.7)
    ax.bar([i + width for i in x], domain_stats['gap_score'], width, label='Gap Score', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_stats['domain'], rotation=45, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Average Scores by Research Domain', fontsize=14)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/domain_analysis.png', dpi=150)
    plt.close()

    print(f"Saved visualizations to {output_dir}/")


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("EXPERIMENT: Can LLMs Expose What Science Refuses to See?")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # Get API client
    print("\n1. Setting up API client...")
    try:
        client = get_openrouter_client()
        model = "openai/gpt-4o"  # Use GPT-4o via OpenRouter
        print(f"   Using model: {model}")
    except ValueError as e:
        print(f"   OpenRouter not available: {e}")
        try:
            client = get_openai_client()
            model = "gpt-4o"
            print(f"   Using OpenAI directly with model: {model}")
        except ValueError as e2:
            print(f"   ERROR: No API key available. {e2}")
            return None

    # Create topics
    print("\n2. Creating research topic set...")
    topics = create_research_topics()
    print(f"   Created {len(topics)} topics across domains: {set(t['domain'] for t in topics)}")

    # Query LLM for scores
    print("\n3. Querying LLM for attention and impact scores...")
    llm_scores = query_llm_for_scores(client, topics, model)
    successful = sum(1 for v in llm_scores.values() if v.get('attention_score') is not None)
    print(f"   Successfully scored {successful}/{len(topics)} topics")

    # Query LLM for direct gap identification
    print("\n4. Querying LLM for direct gap identification...")
    gap_results = query_llm_for_gap_identification(client, model)
    if 'gaps' in gap_results:
        print(f"   LLM identified {len(gap_results['gaps'])} research gaps")
    else:
        print(f"   Gap identification failed: {gap_results.get('error', 'unknown error')}")

    # Analyze results
    print("\n5. Analyzing results...")
    df = analyze_results(topics, llm_scores)
    metrics = compute_validation_metrics(df)

    # Create visualizations
    print("\n6. Creating visualizations...")
    output_dir = "figures"
    create_visualizations(df, output_dir)

    # Save results
    print("\n7. Saving results...")
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'topics': topics,
        'llm_scores': llm_scores,
        'gap_identification': gap_results,
        'metrics': metrics,
        'summary_statistics': {
            'mean_attention': df['llm_attention'].mean(),
            'std_attention': df['llm_attention'].std(),
            'mean_impact': df['llm_impact'].mean(),
            'std_impact': df['llm_impact'].std(),
            'mean_gap': df['gap_score'].mean(),
            'std_gap': df['gap_score'].std()
        }
    }

    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    df.to_csv('results/topic_scores.csv', index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n[H1] Attention Estimation Correlation:")
    print(f"   Spearman r = {metrics['attention_correlation']['spearman_r']:.3f}")
    print(f"   p-value = {metrics['attention_correlation']['p_value']:.4f}")

    print(f"\n[H2] Impact Estimation Correlation:")
    print(f"   Spearman r = {metrics['impact_correlation']['spearman_r']:.3f}")
    print(f"   p-value = {metrics['impact_correlation']['p_value']:.4f}")

    print(f"\n[H3] Gap Identification:")
    gi = metrics['gap_identification']
    if gi.get('mean_gap_score_true') is not None:
        print(f"   True gap topics (n={gi['n_true_gaps']}): mean gap = {gi['mean_gap_score_true']:.2f}")
        print(f"   Other topics (n={gi['n_other']}): mean gap = {gi['mean_gap_score_other']:.2f}")
        if gi.get('p_value') is not None:
            print(f"   t-test: t = {gi['t_statistic']:.3f}, p = {gi['p_value']:.4f}")
            print(f"   Effect size (Cohen's d) = {gi['effect_size_d']:.3f}")

    print(f"\n[Top 5 Identified Gaps - by LLM scores]:")
    top_gaps = df.nlargest(5, 'gap_score')[['name', 'llm_attention', 'llm_impact', 'gap_score']]
    for _, row in top_gaps.iterrows():
        print(f"   - {row['name']}: gap={row['gap_score']:.1f} (attn={row['llm_attention']:.1f}, impact={row['llm_impact']:.1f})")

    if 'gaps' in gap_results:
        print(f"\n[LLM Direct Gap Identification]:")
        for i, gap in enumerate(gap_results['gaps'][:5], 1):
            print(f"   {i}. {gap['topic']}")
            print(f"      Impact: {gap['estimated_impact_score']}/10, Attention: {gap['estimated_attention_score']}/10")

    print(f"\nResults saved to: results/experiment_results.json")
    print(f"Visualizations saved to: figures/")
    print(f"\nCompleted at: {datetime.now().isoformat()}")

    return results


if __name__ == "__main__":
    run_experiment()
