"""
Load and process benchmark pickle files for probing.

Implements behavioral stratification from MI research plan Part 7.
"""

import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

# Add benchmark directory to path so pickle can find morphology, ontology, etc.
_benchmark_dir = Path(__file__).parent.parent / "benchmark"
if str(_benchmark_dir) not in sys.path:
    sys.path.insert(0, str(_benchmark_dir))


# System prompt used in factorial experiments (from run_factorial_experiment.py)
SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


@dataclass
class ProbingExample:
    """Single example for probing experiments."""

    # Identification
    idx: int                       # Pair index
    h1_or_h2: str                  # "h1" or "h2"
    seed: int                      # Reproducibility seed

    # Input text
    prompt: str                    # Full formatted prompt
    theories_nl: str               # Theory sentences
    observations_nl: str           # Observation sentences

    # Concept hierarchy
    child_concept: str             # e.g., "dalpist"
    root_concept: str              # e.g., "rompus"
    entities: List[str]            # e.g., ["Amy", "Bob", "Carol"]

    # Property
    property_name: str             # e.g., "rainy"
    property_family: str           # e.g., "weather"
    is_negated: bool               # Whether property is negated

    # Ground truth
    ground_truth: str              # Expected hypothesis NL
    depth: int                     # 1 for H1, 2 for H2

    # Behavioral results (from factorial experiments)
    model_correct: Optional[bool] = None
    model_response: Optional[str] = None


def load_matched_pairs(
    pkl_path: str = None
) -> List[Tuple[Dict, Dict]]:
    """
    Load matched H1/H2 pairs from pickle file.

    Args:
        pkl_path: Path to matched_pairs pickle file

    Returns:
        List of (h1_example, h2_example) tuples
    """
    if pkl_path is None:
        # Default path relative to this file
        pkl_path = Path(__file__).parent.parent / "benchmark" / "matched_pairs_set1_pure.pkl"

    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Matched pairs file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        pairs = pickle.load(f)

    return pairs


def load_factorial_results(
    pkl_path: str = None
) -> Dict:
    """
    Load Gemma 2 9B factorial experiment results.

    Args:
        pkl_path: Path to factorial results pickle file

    Returns:
        Results dict with structure:
        {
            'set1': {
                'h1_strong': float,
                'h2_strong': float,
                'h1_results': [{'reply': str, 'gt': str, 'strong': 0|1, ...}, ...],
                'h2_results': [{'reply': str, 'gt': str, 'strong': 0|1, ...}, ...],
                'pairs': [...]
            },
            ...
        }
    """
    if pkl_path is None:
        # Default path relative to this file
        pkl_path = Path(__file__).parent.parent / "benchmark" / "factorial_results" / "factorial_gemma29b.pkl"

    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Factorial results file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    return results


def format_prompt(
    theories_nl: str,
    observations_nl: str,
    include_system_prompt: bool = True
) -> str:
    """
    Format prompt for Gemma 2 IT.

    Note: Gemma 2 IT doesn't have a separate system prompt format,
    so we prepend the system prompt to the user message.

    Args:
        theories_nl: Theory sentences
        observations_nl: Observation sentences
        include_system_prompt: Whether to include system prompt

    Returns:
        Formatted prompt string
    """
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."

    if include_system_prompt:
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    else:
        return user_prompt


def create_probing_dataset(
    pairs_path: str = None,
    results_path: str = None,
    set_key: str = "set1",
    include_system_prompt: bool = True
) -> List[ProbingExample]:
    """
    Create probing dataset by combining pairs with behavioral results.

    Args:
        pairs_path: Path to matched pairs pickle
        results_path: Path to factorial results pickle
        set_key: Which set to use (default: "set1")
        include_system_prompt: Whether to include system prompt in prompts

    Returns:
        List of ProbingExample objects annotated with behavioral outcomes
    """
    pairs = load_matched_pairs(pairs_path)
    results = load_factorial_results(results_path)

    examples = []

    set_results = results.get(set_key, {})
    h1_results = set_results.get('h1_results', [])
    h2_results = set_results.get('h2_results', [])

    for idx, (h1, h2) in enumerate(pairs):
        for example, h_type, result_list in [(h1, "h1", h1_results), (h2, "h2", h2_results)]:
            # Format prompt
            prompt = format_prompt(
                example['theories_nl'],
                example['observations_nl'],
                include_system_prompt=include_system_prompt
            )

            # Get behavioral result if available
            result = result_list[idx] if idx < len(result_list) else None

            # Extract property info
            prop = example['property']
            if hasattr(prop, 'name'):
                property_name = prop.name
                property_family = prop.family if hasattr(prop, 'family') else ""
            else:
                property_name = str(prop)
                property_family = ""

            probing_ex = ProbingExample(
                idx=idx,
                h1_or_h2=h_type,
                seed=example['seed'],
                prompt=prompt,
                theories_nl=example['theories_nl'],
                observations_nl=example['observations_nl'],
                child_concept=example['child_concept'],
                root_concept=example['root_concept'],
                entities=example['entities'],
                property_name=property_name,
                property_family=property_family,
                is_negated=example.get('is_negated', False),
                ground_truth=example['gt_hypothesis_nl'],
                depth=example['depth'],
                model_correct=bool(result['strong']) if result else None,
                model_response=result.get('reply', '') if result else None
            )

            examples.append(probing_ex)

    return examples


def stratify_gemma_results(
    results_path: str = None,
    pairs_path: str = None,
    set_key: str = "set1",
    verbose: bool = True
) -> Dict[str, List[int]]:
    """
    Stratify Gemma results into behavioral categories for MI analysis.

    From MI research plan Part 7:
    - h1_fail_h2_success: Parent output for both (GOLD pairs for MI)
    - h1_success_h2_fail: Child output for both (standard behavior)
    - h1_success_h2_success: Task-sensitive (rare, investigate)
    - h1_fail_h2_fail: Neither correct (parsing issues)

    Args:
        results_path: Path to factorial results
        pairs_path: Path to matched pairs
        set_key: Which set to analyze
        verbose: Whether to print summary

    Returns:
        Dict mapping category names to lists of pair indices
    """
    results = load_factorial_results(results_path)

    set_results = results.get(set_key, {})
    h1_results = set_results.get('h1_results', [])
    h2_results = set_results.get('h2_results', [])

    categories = {
        'h1_fail_h2_success': [],  # Parent output for both (GOLD)
        'h1_success_h2_fail': [],  # Child output for both (standard)
        'h1_success_h2_success': [],  # Task-sensitive (rare)
        'h1_fail_h2_fail': [],  # Neither correct
    }

    n_pairs = min(len(h1_results), len(h2_results))

    for i in range(n_pairs):
        h1_success = h1_results[i]['strong'] == 1
        h2_success = h2_results[i]['strong'] == 1

        if not h1_success and h2_success:
            categories['h1_fail_h2_success'].append(i)
        elif h1_success and not h2_success:
            categories['h1_success_h2_fail'].append(i)
        elif h1_success and h2_success:
            categories['h1_success_h2_success'].append(i)
        else:
            categories['h1_fail_h2_fail'].append(i)

    if verbose:
        # Verify Conservation Law
        h1_acc = sum(1 for r in h1_results if r['strong'] == 1) / len(h1_results)
        h2_acc = sum(1 for r in h2_results if r['strong'] == 1) / len(h2_results)

        print("=" * 60)
        print("GEMMA 2 9B BEHAVIORAL STRATIFICATION")
        print("=" * 60)
        for cat, indices in categories.items():
            pct = 100 * len(indices) / n_pairs if n_pairs > 0 else 0
            print(f"{cat}: {len(indices)} ({pct:.1f}%)")
        print(f"\nH1 accuracy: {h1_acc:.1%}")
        print(f"H2 accuracy: {h2_acc:.1%}")
        print(f"Sum: {h1_acc + h2_acc:.1%} (expected: ~100%)")
        print(f"Implied p: {h2_acc:.3f}")
        print("=" * 60)

    return categories


def get_gold_pairs(
    examples: List[ProbingExample] = None,
    results_path: str = None,
    pairs_path: str = None
) -> List[int]:
    """
    Get indices of "gold pairs" - cases where model over-generalizes.

    These are H1 failures + H2 successes (same parent-output behavior,
    but H2 happens to be "correct").

    From CLAUDE.md: Indices [7, 63, 106, 134, 141, ...]

    Args:
        examples: List of ProbingExample (optional, will create if not provided)
        results_path: Path to factorial results
        pairs_path: Path to matched pairs

    Returns:
        List of pair indices for gold pairs
    """
    if examples is None:
        examples = create_probing_dataset(pairs_path, results_path)

    # Group by pair index
    h1_by_idx = {e.idx: e for e in examples if e.h1_or_h2 == "h1"}
    h2_by_idx = {e.idx: e for e in examples if e.h1_or_h2 == "h2"}

    gold_indices = []

    for idx in h1_by_idx:
        h1 = h1_by_idx[idx]
        h2 = h2_by_idx.get(idx)

        if h1 and h2:
            # Gold pair: H1 fail (outputs parent) AND H2 success (outputs parent, which is correct)
            if h1.model_correct == False and h2.model_correct == True:
                gold_indices.append(idx)

    return gold_indices


def get_examples_by_category(
    examples: List[ProbingExample],
    categories: Dict[str, List[int]]
) -> Dict[str, List[ProbingExample]]:
    """
    Group examples by behavioral category.

    Args:
        examples: List of ProbingExample
        categories: Output from stratify_gemma_results()

    Returns:
        Dict mapping category names to lists of ProbingExample
    """
    # Create lookup by (idx, h1_or_h2)
    example_lookup = {(e.idx, e.h1_or_h2): e for e in examples}

    result = {}
    for cat_name, indices in categories.items():
        cat_examples = []
        for idx in indices:
            # Add both H1 and H2 versions
            if (idx, "h1") in example_lookup:
                cat_examples.append(example_lookup[(idx, "h1")])
            if (idx, "h2") in example_lookup:
                cat_examples.append(example_lookup[(idx, "h2")])
        result[cat_name] = cat_examples

    return result
