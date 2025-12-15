"""
Model Comparison Experiment: Test gap identification across multiple LLMs.

This validates findings by comparing GPT-4o with Claude-3.5-Sonnet.
"""

import os
import json
import random
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from scipy import stats
from datetime import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_openrouter_client() -> OpenAI:
    """Get OpenRouter client for model access."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def create_topics() -> List[Dict]:
    """Subset of topics for model comparison."""
    return [
        {"id": "t1", "name": "Large Language Model Pre-training", "domain": "AI/ML", "expected_attention": "high", "expected_impact": "medium"},
        {"id": "t2", "name": "Deep Reinforcement Learning for Games", "domain": "AI/ML", "expected_attention": "high", "expected_impact": "low"},
        {"id": "t3", "name": "Breast Cancer Detection with AI", "domain": "Medicine", "expected_attention": "high", "expected_impact": "high"},
        {"id": "t4", "name": "Drug Discovery with Deep Learning", "domain": "Medicine", "expected_attention": "high", "expected_impact": "high"},
        {"id": "t5", "name": "Gestational Diabetes Mellitus Management", "domain": "Medicine", "expected_attention": "low", "expected_impact": "high"},
        {"id": "t6", "name": "Endometriosis Diagnosis and Treatment", "domain": "Medicine", "expected_attention": "low", "expected_impact": "high"},
        {"id": "t7", "name": "AI for Low-Resource Languages", "domain": "AI/ML", "expected_attention": "low", "expected_impact": "high"},
        {"id": "t8", "name": "Neglected Tropical Diseases Detection", "domain": "Medicine", "expected_attention": "low", "expected_impact": "high"},
        {"id": "t9", "name": "Small-Scale Farmer Decision Support", "domain": "Agriculture", "expected_attention": "low", "expected_impact": "high"},
        {"id": "t10", "name": "Social Media Recommendation Algorithms", "domain": "AI/ML", "expected_attention": "high", "expected_impact": "low"},
    ]


def query_model(client: OpenAI, topics: List[Dict], model: str) -> Dict:
    """Query a single model for topic scores."""
    results = {}

    for topic in tqdm(topics, desc=f"Querying {model}"):
        prompt = f"""For this research topic, provide two scores (1-10):

TOPIC: {topic['name']}

1. RESEARCH ATTENTION: How much academic research attention does this topic receive?
2. SOCIETAL IMPACT: How important is this for human well-being?

Respond in JSON: {{"attention": <1-10>, "impact": <1-10>}}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content)
            results[topic['id']] = {
                'attention': float(parsed['attention']),
                'impact': float(parsed['impact']),
                'gap': float(parsed['impact']) - float(parsed['attention'])
            }
        except Exception as e:
            print(f"Error: {e}")
            results[topic['id']] = {'error': str(e)}

    return results


def compute_agreement(results1: Dict, results2: Dict) -> Dict:
    """Compute agreement between two models."""
    attention1, attention2 = [], []
    impact1, impact2 = [], []
    gap1, gap2 = [], []

    for tid in results1:
        if 'attention' in results1[tid] and 'attention' in results2.get(tid, {}):
            attention1.append(results1[tid]['attention'])
            attention2.append(results2[tid]['attention'])
            impact1.append(results1[tid]['impact'])
            impact2.append(results2[tid]['impact'])
            gap1.append(results1[tid]['gap'])
            gap2.append(results2[tid]['gap'])

    if len(attention1) < 3:
        return {'error': 'Insufficient data'}

    return {
        'n_topics': len(attention1),
        'attention_correlation': {
            'pearson_r': stats.pearsonr(attention1, attention2)[0],
            'p_value': stats.pearsonr(attention1, attention2)[1]
        },
        'impact_correlation': {
            'pearson_r': stats.pearsonr(impact1, impact2)[0],
            'p_value': stats.pearsonr(impact1, impact2)[1]
        },
        'gap_correlation': {
            'pearson_r': stats.pearsonr(gap1, gap2)[0],
            'p_value': stats.pearsonr(gap1, gap2)[1]
        },
        'mean_attention_diff': np.mean(np.abs(np.array(attention1) - np.array(attention2))),
        'mean_impact_diff': np.mean(np.abs(np.array(impact1) - np.array(impact2))),
        # Check if same topics are identified as top gaps
        'top_gap_agreement': calculate_top_gap_agreement(gap1, gap2, n=3)
    }


def calculate_top_gap_agreement(gap1: List, gap2: List, n: int = 3) -> float:
    """Calculate what fraction of top-n gaps are shared between models."""
    idx1 = set(np.argsort(gap1)[-n:])
    idx2 = set(np.argsort(gap2)[-n:])
    return len(idx1 & idx2) / n


def run_comparison():
    """Run model comparison experiment."""
    print("=" * 60)
    print("MODEL COMPARISON: Cross-model validation of gap identification")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")

    client = get_openrouter_client()
    topics = create_topics()

    # Test two models
    models = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet"
    ]

    all_results = {}
    for model in models:
        print(f"\nQuerying {model}...")
        all_results[model] = query_model(client, topics, model)

    # Compute agreement
    print("\nComputing cross-model agreement...")
    agreement = compute_agreement(
        all_results["openai/gpt-4o"],
        all_results["anthropic/claude-3.5-sonnet"]
    )

    # Print results
    print("\n" + "=" * 60)
    print("CROSS-MODEL AGREEMENT RESULTS")
    print("=" * 60)

    if 'error' not in agreement:
        print(f"\nTopics analyzed: {agreement['n_topics']}")
        print(f"\nAttention Score Correlation:")
        print(f"   Pearson r = {agreement['attention_correlation']['pearson_r']:.3f}")
        print(f"   p-value = {agreement['attention_correlation']['p_value']:.4f}")

        print(f"\nImpact Score Correlation:")
        print(f"   Pearson r = {agreement['impact_correlation']['pearson_r']:.3f}")
        print(f"   p-value = {agreement['impact_correlation']['p_value']:.4f}")

        print(f"\nGap Score Correlation:")
        print(f"   Pearson r = {agreement['gap_correlation']['pearson_r']:.3f}")
        print(f"   p-value = {agreement['gap_correlation']['p_value']:.4f}")

        print(f"\nMean absolute difference in attention scores: {agreement['mean_attention_diff']:.2f}")
        print(f"Mean absolute difference in impact scores: {agreement['mean_impact_diff']:.2f}")
        print(f"Top-3 gap agreement: {agreement['top_gap_agreement']*100:.0f}%")

    # Display per-topic comparison
    print("\nPer-topic comparison:")
    print("-" * 80)
    print(f"{'Topic':<40} {'GPT-4o Gap':>12} {'Claude Gap':>12}")
    print("-" * 80)

    for topic in topics:
        gpt_result = all_results["openai/gpt-4o"].get(topic['id'], {})
        claude_result = all_results["anthropic/claude-3.5-sonnet"].get(topic['id'], {})
        gpt_gap = gpt_result.get('gap', 'N/A')
        claude_gap = claude_result.get('gap', 'N/A')
        gpt_gap_str = f"{gpt_gap:.1f}" if isinstance(gpt_gap, (int, float)) else gpt_gap
        claude_gap_str = f"{claude_gap:.1f}" if isinstance(claude_gap, (int, float)) else claude_gap
        print(f"{topic['name']:<40} {gpt_gap_str:>12} {claude_gap_str:>12}")

    # Save results
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'models': models,
        'topics': topics,
        'model_results': all_results,
        'agreement_metrics': agreement
    }

    with open('results/model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\nResults saved to: results/model_comparison.json")
    print(f"Completed at: {datetime.now().isoformat()}")

    return comparison_results


if __name__ == "__main__":
    run_comparison()
