"""
Evaluation script for INABHYD experiments.
Computes weak accuracy, strong accuracy, and hypothesis quality.

Based on the paper's definitions:
- Strong accuracy: Predicted hypotheses exactly match ground truth
- Weak accuracy: Predicted hypotheses + theories can logically derive all observations
- Quality: q(H) = avg(n(h)) / avg(n(h*)) where n(h) = proof tree appearances
"""

import pickle
import re
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


# =============================================================================
# KNOWLEDGE BASE AND LOGICAL INFERENCE
# =============================================================================

# Known properties from morphology.py - these are adjectives, not concepts
KNOWN_PROPERTIES = {
    # color
    "blue", "red", "brown", "orange",
    # size
    "small", "large",
    # material
    "metallic", "wooden", "luminous", "liquid",
    # light
    "transparent", "opaque", "translucent",
    # mood
    "nervous", "happy", "feisty", "shy", "sad",
    # meta_color
    "bright", "dull", "dark", "pale",
    # taste
    "sweet", "sour", "spicy", "bitter", "salty",
    # perfume
    "floral", "fruity", "earthy", "oriental",
    # temperature
    "hot", "cold", "temperate",
    # personality
    "kind", "mean", "angry", "amenable", "aggressive",
    # sound
    "melodic", "muffled", "discordant", "loud",
    # speed
    "slow", "moderate", "fast",
    # weather
    "windy", "sunny", "overcast", "rainy", "snowy"
}

# Known entity names from morphology.py - these are proper nouns
KNOWN_ENTITIES = {
    "james", "mary", "michael", "patricia", "robert", "jennifer",
    "john", "linda", "david", "elizabeth", "william", "barbara",
    "richard", "susan", "joseph", "jessica", "thomas", "karen",
    "christopher", "sarah", "charles", "lisa", "daniel", "nancy",
    "matthew", "sandra", "anthony", "betty", "mark", "ashley",
    "donald", "emily", "steven", "kimberly", "andrew", "margaret",
    "paul", "donna", "joshua", "michelle", "kenneth", "carol",
    "kevin", "amanda", "brian", "melissa", "timothy", "deborah",
    "ronald", "stephanie", "george", "rebecca", "jason", "sharon",
    "edward", "laura", "jeffrey", "cynthia", "ryan", "dorothy",
    "jacob", "amy", "nicholas", "kathleen", "gary", "angela",
    "eric", "shirley", "jonathan", "emma", "stephen", "brenda",
    "larry", "pamela", "justin", "nicole", "scott", "anna",
    "brandon", "samantha", "gregory", "debra", "alexander", "rachel",
    "patrick", "carolyn", "frank", "janet", "raymond", "maria",
    "jack", "olivia", "dennis", "heather", "jerry", "helen"
}


class KnowledgeBase:
    """
    A simple knowledge base for FOL inference in the INABHYD benchmark.

    Supports three types of facts:
    1. Membership: "Entity is a Concept" (e.g., "Sam is a cat")
    2. Inheritance: "Concept is a ParentConcept" (e.g., "each cat is a mammal")
    3. Property: "Concept/Entity is Property" (e.g., "each mammal is hairy")
    4. Negated Property: "Concept/Entity is not Property" (e.g., "felines are not slow")

    Inference rules:
    - If X is a Y, and Y is a Z, then X is a Z (transitivity)
    - If X is a Y, and Y is P (property), then X is P (property inheritance)
    - If X is a Y, and Y is not P, then X is not P (negated property inheritance)
    """

    def __init__(self):
        # membership[entity] = set of concepts the entity belongs to
        self.membership = defaultdict(set)
        # inheritance[concept] = set of parent concepts
        self.inheritance = defaultdict(set)
        # properties[concept] = set of properties the concept has (positive)
        self.properties = defaultdict(set)
        # negated_properties[concept] = set of properties the concept does NOT have
        self.negated_properties = defaultdict(set)
        # All facts as (subject, predicate, is_negated) tuples
        self.facts = set()

    def _is_property(self, word):
        """Check if a word is a known property (adjective)."""
        return word.lower() in KNOWN_PROPERTIES

    def _is_entity(self, word):
        """Check if a word is a known entity (proper noun)."""
        return word.lower() in KNOWN_ENTITIES

    def add_fact(self, subject, predicate, is_negated=False):
        """Add a fact to the knowledge base with proper type distinction."""
        subject = normalize_to_singular(subject.lower().strip())
        predicate = normalize_to_singular(predicate.lower().strip())

        # Handle negation in predicate
        if predicate.startswith('not '):
            predicate = predicate[4:]
            is_negated = True

        self.facts.add((subject, predicate, is_negated))

        # Determine fact type and store appropriately
        if is_negated:
            # Negated facts are always properties
            self.negated_properties[subject].add(predicate)
        elif self._is_property(predicate):
            # Predicate is a known property adjective
            self.properties[subject].add(predicate)
        elif self._is_entity(subject):
            # Subject is an entity, predicate is a concept -> membership
            self.membership[subject].add(predicate)
        else:
            # Subject is a concept, predicate is a concept -> inheritance
            self.inheritance[subject].add(predicate)

    def add_from_text(self, text):
        """Parse and add facts from natural language text."""
        for sent in text.replace('.', '. ').split('.'):
            sent = sent.strip()
            if not sent:
                continue

            struct = parse_hypothesis_structure(sent)
            if struct:
                subj, pred = struct
                is_negated = pred.startswith('not ')
                if is_negated:
                    pred = pred[4:]
                self.add_fact(subj, pred, is_negated)

    def get_all_concepts_for_entity(self, entity):
        """
        Get all concepts an entity belongs to, following inheritance chains.
        Returns set of (concept, proof_depth) tuples.
        """
        entity = normalize_to_singular(entity.lower().strip())
        result = set()
        visited = set()
        queue = [(c, 1) for c in self.membership.get(entity, set())]

        while queue:
            concept, depth = queue.pop(0)
            if concept in visited:
                continue
            visited.add(concept)
            result.add((concept, depth))

            # Follow inheritance chain
            for parent in self.inheritance.get(concept, set()):
                if parent not in visited:
                    queue.append((parent, depth + 1))

        return result

    def get_all_properties_for_entity(self, entity, include_negated=True):
        """
        Get all properties an entity has, following concept inheritance.

        Returns:
            If include_negated=True: (positive_props, negated_props) where each is set of (prop, depth)
            If include_negated=False: set of (prop, depth) for positive properties only
        """
        entity = normalize_to_singular(entity.lower().strip())
        positive_result = set()
        negated_result = set()

        # Direct properties of the entity
        for prop in self.properties.get(entity, set()):
            positive_result.add((prop, 1))
        for prop in self.negated_properties.get(entity, set()):
            negated_result.add((prop, 1))

        # Properties inherited through concepts
        for concept, concept_depth in self.get_all_concepts_for_entity(entity):
            for prop in self.properties.get(concept, set()):
                positive_result.add((prop, concept_depth + 1))
            for prop in self.negated_properties.get(concept, set()):
                negated_result.add((prop, concept_depth + 1))

        if include_negated:
            return positive_result, negated_result
        return positive_result

    def can_derive(self, subject, predicate, is_negated=False):
        """
        Check if we can derive that subject has predicate.
        Returns (can_derive: bool, proof_depth: int or None)
        """
        subject = normalize_to_singular(subject.lower().strip())
        predicate = normalize_to_singular(predicate.lower().strip())

        # Handle negation in predicate string
        if predicate.startswith('not '):
            predicate = predicate[4:]
            is_negated = True

        # Check direct facts
        if (subject, predicate, is_negated) in self.facts:
            return True, 1

        # Get all properties (positive and negated)
        positive_props, negated_props = self.get_all_properties_for_entity(subject, include_negated=True)

        if is_negated:
            # Looking for negated property - check through inheritance chain
            for prop, depth in negated_props:
                if prop == predicate:
                    return True, depth
        else:
            # Looking for positive property OR concept membership
            # First check if predicate is a concept the entity belongs to
            for concept, depth in self.get_all_concepts_for_entity(subject):
                if concept == predicate:
                    return True, depth

            # Then check if predicate is a positive property
            for prop, depth in positive_props:
                if prop == predicate:
                    return True, depth

        return False, None


# =============================================================================
# TEXT NORMALIZATION AND PARSING
# =============================================================================

def normalize_to_singular(word):
    """
    Convert a potentially plural word to singular.
    Handles the benchmark's plural rules from fol.py:
    - If singular ends in 's', plural is word + 'es' (e.g., "rompus" -> "rompuses")
    - Otherwise, plural is word + 's'
    """
    word = word.strip()
    if len(word) <= 2:
        return word

    # Handle 'es' suffix for words where singular ends in s, x, z, ch, sh
    if word.endswith('ses') and len(word) > 3:
        return word[:-2]
    if word.endswith('xes') and len(word) > 3:
        return word[:-2]
    if word.endswith('zes') and len(word) > 3:
        return word[:-2]
    if word.endswith('ches') and len(word) > 4:
        return word[:-2]
    if word.endswith('shes') and len(word) > 4:
        return word[:-2]

    # Don't strip 's' from words that end in common singular patterns
    if word.endswith('us') or word.endswith('is') or word.endswith('os'):
        return word

    # Regular plural - just remove 's'
    if word.endswith('s') and len(word) > 2:
        return word[:-1]

    return word


def normalize_hypothesis(hyp):
    """
    Normalize a hypothesis string for comparison.
    Handles FOL variations like "each X is Y", "every X is Y", "all X are Y", etc.
    """
    hyp = hyp.lower().strip()
    # Remove punctuation at the end
    hyp = re.sub(r'[.!?,;:]+$', '', hyp)
    # Normalize whitespace
    hyp = ' '.join(hyp.split())

    # Normalize FOL quantifiers to a standard form
    hyp = re.sub(r'^(each|every|all)\s+', '', hyp)
    hyp = re.sub(r'\s+are\s+', ' is ', hyp)

    # Remove articles
    hyp = re.sub(r'\s+(a|an|the)\s+', ' ', hyp)
    hyp = re.sub(r'^(a|an|the)\s+', '', hyp)

    # Normalize to singular forms for comparison
    parts = hyp.split(' is ')
    if len(parts) == 2:
        subj = normalize_to_singular(parts[0])
        obj = normalize_to_singular(parts[1])
        hyp = f"{subj} is {obj}"

    return hyp


def parse_hypothesis_structure(hyp):
    """
    Parse a hypothesis into (subject, predicate) structure.
    Returns (subject, predicate) tuple or None if unparseable.
    """
    norm = normalize_hypothesis(hyp)

    # Match "X is not Y" pattern first
    match = re.match(r'^(.+?)\s+is\s+not\s+(.+)$', norm)
    if match:
        return (match.group(1).strip(), f"not {match.group(2).strip()}")

    # Match "X is Y" pattern
    match = re.match(r'^(.+?)\s+is\s+(.+)$', norm)
    if match:
        return (match.group(1).strip(), match.group(2).strip())

    return None


def extract_after_thinking(response):
    """
    Extract content after thinking tags for Qwen3-style models.
    Returns (content, had_thinking) tuple.
    - If </think> exists, return content after it
    - If <think> exists but no </think>, output was truncated - return None
    - If no <think>, return original response
    """
    if '<think>' not in response:
        return response, False

    # Check if thinking is complete
    if '</think>' in response:
        # Extract content after </think>
        idx = response.rfind('</think>')
        return response[idx + 8:].strip(), True
    else:
        # Truncated output - thinking not complete
        return None, True


def parse_hypotheses_from_response(response):
    """Extract hypotheses from LLM response."""
    if not response:
        return []

    hypotheses = []

    # Handle thinking tags (Qwen3, R1 models)
    content, had_thinking = extract_after_thinking(response)
    if content is None:
        # Truncated thinking - no answer available
        return []
    response = content if content else response

    # Also remove any remaining thinking tags (legacy behavior)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Split by newlines and filter
    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#') or (line.startswith('*') and len(line) < 5):
            continue

        # Skip lines that look like explanations or headers
        skip_patterns = [
            r'^(based on|here are|the following|my hypothes|final hypothes|to explain)',
            r'^(observation|therefore|thus|so|because|since|given)',
            r'hypothes[ie]s?.*:',
            r'^\*\*',
        ]
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, line.lower()):
                should_skip = True
                break
        if should_skip:
            continue

        # Check if line contains "is" pattern (hypothesis format)
        if ' is ' in line.lower() or ' are ' in line.lower():
            # Remove bullet points, numbers, etc.
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line)
            line = re.sub(r'^hypothesis\s*\d*\s*:?\s*', '', line, flags=re.IGNORECASE)
            line = re.sub(r'^(final\s+)?hypothes[ie]s?\s*:?\s*', '', line, flags=re.IGNORECASE)

            # Extract hypothesis before explanation words (because, since, as, etc.)
            # e.g., "Kevin is salty because he is a gwompant" -> "Kevin is salty"
            for sep in [' because ', ' since ', ' as ', ' given ', ' due to ']:
                if sep in line.lower():
                    idx = line.lower().find(sep)
                    line = line[:idx].strip()
                    break

            if line:
                struct = parse_hypothesis_structure(line)
                if struct:
                    subj, pred = struct
                    if len(subj.split()) <= 3 and len(pred.split()) <= 3:
                        hypotheses.append(line)

    return hypotheses


def parse_ground_truth(gt_string):
    """Parse ground truth hypotheses from ontology.hypotheses string."""
    if not gt_string:
        return []
    hypotheses = []
    for h in gt_string.split('.'):
        h = h.strip()
        if h and parse_hypothesis_structure(h):
            hypotheses.append(h)
    return hypotheses


def parse_observations(obs_string):
    """Parse observations from ontology.observations string."""
    if not obs_string:
        return []
    observations = []
    for o in obs_string.split('.'):
        o = o.strip()
        if o and parse_hypothesis_structure(o):
            observations.append(o)
    return observations


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_strong_accuracy(pred_hypotheses, gt_hypotheses, first_only=True):
    """
    Strong accuracy: Predicted hypotheses match ground truth.

    Args:
        pred_hypotheses: List of predicted hypothesis strings
        gt_hypotheses: List of ground truth hypothesis strings
        first_only: If True (default), only compare first hypothesis from each list.
                   This matches the paper's behavior where models often output
                   multiple hypotheses but only the first one is evaluated.
                   If False, requires exact set match.

    Returns 1 if match, 0 otherwise.
    """
    if not pred_hypotheses or not gt_hypotheses:
        return 0

    if first_only:
        # Paper-compatible: only compare first hypothesis
        pred_struct = parse_hypothesis_structure(pred_hypotheses[0])
        gt_struct = parse_hypothesis_structure(gt_hypotheses[0])
        return 1 if pred_struct and gt_struct and pred_struct == gt_struct else 0
    else:
        # Strict: require exact set match
        if len(pred_hypotheses) != len(gt_hypotheses):
            return 0

        pred_set = set()
        for p in pred_hypotheses:
            struct = parse_hypothesis_structure(p)
            if struct:
                pred_set.add(struct)

        gt_set = set()
        for g in gt_hypotheses:
            struct = parse_hypothesis_structure(g)
            if struct:
                gt_set.add(struct)

        return 1 if pred_set == gt_set else 0


def compute_weak_accuracy(pred_hypotheses, gt_hypotheses, observations, theories):
    """
    Weak accuracy: Predicted hypotheses + theories can logically derive all observations.

    This implements proper logical inference:
    1. Build a knowledge base from theories and predicted hypotheses
    2. For each observation, check if it can be derived
    3. Return 1 if ALL observations can be derived, 0 otherwise
    """
    if not pred_hypotheses:
        return 0

    # Parse observations
    obs_list = parse_observations(observations)
    if not obs_list:
        return 0

    # Build knowledge base with theories and predicted hypotheses
    kb = KnowledgeBase()
    kb.add_from_text(theories)

    for hyp in pred_hypotheses:
        struct = parse_hypothesis_structure(hyp)
        if struct:
            subj, pred = struct
            is_negated = pred.startswith('not ')
            if is_negated:
                pred = pred[4:]
            kb.add_fact(subj, pred, is_negated)

    # Check if each observation can be derived
    for obs in obs_list:
        struct = parse_hypothesis_structure(obs)
        if struct:
            subj, pred = struct
            can_derive, _ = kb.can_derive(subj, pred)
            if not can_derive:
                return 0

    return 1


def compute_quality(pred_hypotheses, gt_hypotheses, observations, theories):
    """
    Compute hypothesis quality based on Occam's Razor.
    q(H) = avg(n(h) for h in H) / avg(n(h*) for h* in H*)

    where n(h) = number of observations in whose proof tree h appears

    A hypothesis appears in the proof tree of an observation if:
    - The observation is derivable WITH the hypothesis but NOT without it
    """
    if not pred_hypotheses:
        return 0.0

    # Parse observations
    obs_list = parse_observations(observations)
    if not obs_list:
        return 0.0

    def count_proof_appearances(hypotheses, obs_list, theories):
        """
        For each hypothesis, count how many observations require it in their proof.
        Uses the expert's recommended approach: compare derivability with/without hypothesis.
        """
        counts = []

        # Build base KB with just theories
        kb_base = KnowledgeBase()
        kb_base.add_from_text(theories)

        for hyp in hypotheses:
            hyp_struct = parse_hypothesis_structure(hyp)
            if not hyp_struct:
                counts.append(0)
                continue

            # Build KB with this hypothesis added
            kb_with_hyp = KnowledgeBase()
            kb_with_hyp.add_from_text(theories)

            subj, pred = hyp_struct
            is_negated = pred.startswith('not ')
            if is_negated:
                pred = pred[4:]
            kb_with_hyp.add_fact(subj, pred, is_negated)

            count = 0
            for obs in obs_list:
                obs_struct = parse_hypothesis_structure(obs)
                if not obs_struct:
                    continue

                obs_subj, obs_pred = obs_struct

                # Check if observation is derivable WITH hypothesis but NOT without
                can_with, _ = kb_with_hyp.can_derive(obs_subj, obs_pred)
                can_without, _ = kb_base.can_derive(obs_subj, obs_pred)

                if can_with and not can_without:
                    # Hypothesis is needed to derive this observation
                    count += 1

            counts.append(max(count, 1))  # At least 1 to avoid division issues

        return counts

    # Count proof appearances for predictions and ground truth
    pred_counts = count_proof_appearances(pred_hypotheses, obs_list, theories)
    gt_counts = count_proof_appearances(gt_hypotheses, obs_list, theories)

    # Compute averages
    pred_avg = sum(pred_counts) / len(pred_counts) if pred_counts else 0
    gt_avg = sum(gt_counts) / len(gt_counts) if gt_counts else 1

    if gt_avg == 0:
        gt_avg = 1

    # Quality ratio
    quality = pred_avg / gt_avg

    # Normalize to [0, 1]
    quality = min(1.0, max(0.0, quality))

    return quality


def evaluate_single_example(response, ontology, verbose=False):
    """Evaluate a single example."""
    pred_hypotheses = parse_hypotheses_from_response(response)
    gt_hypotheses = parse_ground_truth(ontology.hypotheses)

    strong_acc = compute_strong_accuracy(pred_hypotheses, gt_hypotheses)
    weak_acc = compute_weak_accuracy(
        pred_hypotheses, gt_hypotheses,
        ontology.observations, ontology.theories
    )
    quality = compute_quality(pred_hypotheses, gt_hypotheses, ontology.observations, ontology.theories)

    # If strong accuracy is 1, weak must be 1 and quality should be 1
    if strong_acc == 1:
        weak_acc = 1
        quality = 1.0

    if verbose:
        print(f"  GT hypotheses: {gt_hypotheses}")
        print(f"  Pred hypotheses: {pred_hypotheses}")
        print(f"  Strong: {strong_acc}, Weak: {weak_acc}, Quality: {quality:.3f}")

    return {
        'strong_accuracy': strong_acc,
        'weak_accuracy': weak_acc,
        'quality': quality,
        'pred_hypotheses': pred_hypotheses,
        'gt_hypotheses': gt_hypotheses
    }


def wilson_confidence_interval(p, n, z=1.96):
    """Compute Wilson score interval for proportion p with n samples."""
    if n == 0:
        return (0, 0)

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    spread = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator

    return (max(0, center - spread), min(1, center + spread))


def evaluate_experiment(replies_file, examples_file=None):
    """Evaluate all examples in an experiment."""
    with open(replies_file, 'rb') as f:
        replies = pickle.load(f)

    if examples_file is None:
        base_name = str(replies_file).replace('_reply_gpt.pkl', '.pkl')
        base_name = base_name.replace('_reply_llama.pkl', '.pkl')
        base_name = base_name.replace('_reply_deepseek.pkl', '.pkl')
        base_name = base_name.replace('_reply_gemmi.pkl', '.pkl')
        examples_file = Path(base_name)

    results = {
        'strong_accuracy': [],
        'weak_accuracy': [],
        'quality': []
    }

    print(f"Evaluating {len(replies)} examples from {replies_file}")

    for i, reply in enumerate(replies):
        pred_hypotheses = parse_hypotheses_from_response(reply)
        results['pred_hypotheses_count'] = results.get('pred_hypotheses_count', [])
        results['pred_hypotheses_count'].append(len(pred_hypotheses))

        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Response preview: {reply[:200]}...")
            print(f"  Parsed hypotheses: {pred_hypotheses}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate INABHYD experiments')
    parser.add_argument('--replies', '-r', type=str, required=True,
                        help='Path to replies pickle file')
    parser.add_argument('--examples', '-e', type=str, default=None,
                        help='Path to examples pickle file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed results')
    args = parser.parse_args()

    results = evaluate_experiment(args.replies, args.examples)

    if 'strong_accuracy' in results and results['strong_accuracy']:
        n = len(results['strong_accuracy'])
        strong_mean = np.mean(results['strong_accuracy'])
        weak_mean = np.mean(results['weak_accuracy'])
        quality_mean = np.mean(results['quality'])

        strong_ci = wilson_confidence_interval(strong_mean, n)
        weak_ci = wilson_confidence_interval(weak_mean, n)

        print(f"\n=== Results (n={n}) ===")
        print(f"Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
        print(f"Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")
        print(f"Quality: {quality_mean:.3f}")
    else:
        print("\n=== Parsing Results ===")
        if 'pred_hypotheses_count' in results:
            print(f"Average hypotheses per response: {np.mean(results['pred_hypotheses_count']):.2f}")


# =============================================================================
# FOL (FIRST-ORDER LOGIC) PARSING AND EVALUATION
# =============================================================================

def parse_fol_hypothesis(text):
    """
    Parse a FOL hypothesis string.
    Returns tuple: (type, subject, predicate, negated) or None

    Examples:
    - "rainy(Amy)" → ("ground", "Amy", "rainy", False)
    - "¬slow(Amy)" → ("ground", "Amy", "slow", True)
    - "dalpist(Amy)" → ("ground", "Amy", "dalpist", False)
    - "∀x(dalpist(x) → rainy(x))" → ("universal", "dalpist", "rainy", False)
    - "∀x(dalpist(x) → ¬slow(x))" → ("universal", "dalpist", "slow", True)
    - "∀x(cat(x) → mammal(x))" → ("universal", "cat", "mammal", False)
    """
    text = text.strip()

    # Handle alternative notations
    text = text.replace('forall ', '∀')
    text = text.replace('forall', '∀')
    text = text.replace(' -> ', '→')
    text = text.replace('->', '→')
    text = text.replace('~', '¬')
    text = text.replace('!', '¬')
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Universal quantifier: ∀x(P(x) → Q(x)) or ∀x(P(x) → ¬Q(x))
    # Also handle ∀ x ( ... ) with spaces
    universal_match = re.match(r'∀\s*x\s*\(\s*(\w+)\s*\(\s*x\s*\)\s*→\s*(¬?)\s*(\w+)\s*\(\s*x\s*\)\s*\)', text)
    if universal_match:
        subj = universal_match.group(1)
        negated = universal_match.group(2) == '¬'
        pred = universal_match.group(3)
        return ("universal", subj, pred, negated)

    # Ground atom: predicate(constant) or ¬predicate(constant)
    ground_match = re.match(r'(¬?)\s*(\w+)\s*\(\s*(\w+)\s*\)', text)
    if ground_match:
        negated = ground_match.group(1) == '¬'
        pred = ground_match.group(2)
        const = ground_match.group(3)
        return ("ground", const, pred, negated)

    return None


def parse_fol_hypotheses_from_response(response):
    """Parse multiple FOL hypotheses from model response."""
    if not response:
        return []

    hypotheses = []

    # Handle thinking tags (Qwen3, R1 models)
    content, had_thinking = extract_after_thinking(response)
    if content is None:
        # Truncated thinking - no answer available
        return []
    response = content if content else response

    # Also remove any remaining thinking tags (legacy behavior)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Split by newlines, periods, semicolons
    lines = re.split(r'[\n;]', response)

    for line in lines:
        # Also split by periods but be careful not to split within parentheses
        parts = re.split(r'\.\s*(?![^(]*\))', line)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Skip explanation lines
            skip_patterns = [
                r'^(based on|here are|the following|my hypothes|final hypothes|to explain)',
                r'^(observation|therefore|thus|so|because|since|given)',
                r'hypothes[ie]s?.*:',
                r'^\*\*',
                r'^theories?:',
                r'^observations?:',
            ]
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, part.lower()):
                    should_skip = True
                    break
            if should_skip:
                continue

            # Remove bullet points, numbers, etc.
            part = re.sub(r'^[\d\.\-\*\•]+\s*', '', part)
            part = re.sub(r'^hypothesis\s*\d*\s*:?\s*', '', part, flags=re.IGNORECASE)

            # Try to parse as FOL
            parsed = parse_fol_hypothesis(part)
            if parsed:
                hypotheses.append(part)

    return hypotheses


def parse_fol_ground_truth(gt_string):
    """Parse FOL ground truth hypotheses from ontology.fol_hypotheses string."""
    if not gt_string:
        return []
    hypotheses = []
    # Split by period but be careful not to split within parentheses
    parts = re.split(r'\.\s*(?![^(]*\))', gt_string)
    for h in parts:
        h = h.strip()
        if h and parse_fol_hypothesis(h):
            hypotheses.append(h)
    return hypotheses


def normalize_fol_hypothesis(hyp):
    """Normalize a FOL hypothesis for comparison."""
    parsed = parse_fol_hypothesis(hyp)
    if parsed:
        hyp_type, subj, pred, negated = parsed
        # Normalize to lowercase
        subj = subj.lower()
        pred = pred.lower()
        return (hyp_type, subj, pred, negated)
    return None


def compute_fol_strong_accuracy(pred_hypotheses, gt_hypotheses, first_only=True):
    """
    Strong accuracy for FOL: Predicted hypotheses match ground truth.

    Args:
        pred_hypotheses: List of predicted FOL hypothesis strings
        gt_hypotheses: List of ground truth FOL hypothesis strings
        first_only: If True (default), only compare first hypothesis from each list.

    Returns 1 if match, 0 otherwise.
    """
    if not pred_hypotheses or not gt_hypotheses:
        return 0

    if first_only:
        # Paper-compatible: only compare first hypothesis
        pred_norm = normalize_fol_hypothesis(pred_hypotheses[0])
        gt_norm = normalize_fol_hypothesis(gt_hypotheses[0])
        return 1 if pred_norm and gt_norm and pred_norm == gt_norm else 0
    else:
        # Strict: require exact set match
        if len(pred_hypotheses) != len(gt_hypotheses):
            return 0

        pred_set = set()
        for p in pred_hypotheses:
            norm = normalize_fol_hypothesis(p)
            if norm:
                pred_set.add(norm)

        gt_set = set()
        for g in gt_hypotheses:
            norm = normalize_fol_hypothesis(g)
            if norm:
                gt_set.add(norm)

        return 1 if pred_set == gt_set else 0


class FOLKnowledgeBase:
    """
    A knowledge base for FOL inference using symbolic FOL format.

    Supports:
    - Ground atoms: predicate(constant), e.g., dalpist(Amy), rainy(Amy)
    - Universal rules: ∀x(P(x) → Q(x)), e.g., ∀x(dalpist(x) → rainy(x))
    - Negation: ¬predicate(constant), ∀x(P(x) → ¬Q(x))
    """

    def __init__(self):
        # ground_atoms[(const, pred)] = True/False (True = positive, False = negated)
        self.ground_atoms = {}
        # universal_rules[(subj_pred, obj_pred)] = True/False
        self.universal_rules = {}
        # membership[const] = set of predicates the constant belongs to
        self.membership = defaultdict(set)
        # properties[pred] = set of properties/parent predicates
        self.properties = defaultdict(set)
        # negated_properties[pred] = set of negated properties
        self.negated_properties = defaultdict(set)

    def add_from_fol(self, fol_text):
        """Parse and add facts from FOL text."""
        # Split by period but be careful not to split within parentheses
        parts = re.split(r'\.\s*(?![^(]*\))', fol_text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            parsed = parse_fol_hypothesis(part)
            if parsed:
                hyp_type, subj, pred, negated = parsed
                subj = subj.lower()
                pred = pred.lower()

                if hyp_type == "ground":
                    # subj is constant, pred is predicate
                    self.ground_atoms[(subj, pred)] = not negated
                    if not negated:
                        self.membership[subj].add(pred)
                elif hyp_type == "universal":
                    # subj is antecedent predicate, pred is consequent predicate
                    self.universal_rules[(subj, pred)] = not negated
                    if negated:
                        self.negated_properties[subj].add(pred)
                    else:
                        self.properties[subj].add(pred)

    def get_all_predicates_for_constant(self, const):
        """Get all predicates a constant has, following inference chains."""
        const = const.lower()
        result = set()
        visited = set()
        queue = list(self.membership.get(const, set()))

        while queue:
            pred = queue.pop(0)
            if pred in visited:
                continue
            visited.add(pred)
            result.add((pred, False))  # (predicate, is_negated)

            # Follow universal rules
            for parent_pred in self.properties.get(pred, set()):
                if parent_pred not in visited:
                    queue.append(parent_pred)
            for neg_pred in self.negated_properties.get(pred, set()):
                result.add((neg_pred, True))

        return result

    def can_derive(self, const, pred, negated=False):
        """Check if we can derive pred(const) or ¬pred(const)."""
        const = const.lower()
        pred = pred.lower()

        # Check direct facts
        if (const, pred) in self.ground_atoms:
            is_positive = self.ground_atoms[(const, pred)]
            if negated and not is_positive:
                return True
            if not negated and is_positive:
                return True

        # Check through inference
        all_preds = self.get_all_predicates_for_constant(const)
        for p, is_neg in all_preds:
            if p == pred and is_neg == negated:
                return True

        return False


def compute_fol_weak_accuracy(pred_hypotheses, gt_hypotheses, fol_observations, fol_theories):
    """
    Weak accuracy for FOL: Predicted hypotheses + theories can logically derive all observations.
    """
    if not pred_hypotheses:
        return 0

    # Parse observations
    obs_list = parse_fol_ground_truth(fol_observations)
    if not obs_list:
        return 0

    # Build knowledge base with theories and predicted hypotheses
    kb = FOLKnowledgeBase()
    kb.add_from_fol(fol_theories)

    for hyp in pred_hypotheses:
        parsed = parse_fol_hypothesis(hyp)
        if parsed:
            hyp_type, subj, pred, negated = parsed
            if hyp_type == "ground":
                kb.ground_atoms[(subj.lower(), pred.lower())] = not negated
                if not negated:
                    kb.membership[subj.lower()].add(pred.lower())
            elif hyp_type == "universal":
                kb.universal_rules[(subj.lower(), pred.lower())] = not negated
                if negated:
                    kb.negated_properties[subj.lower()].add(pred.lower())
                else:
                    kb.properties[subj.lower()].add(pred.lower())

    # Check if each observation can be derived
    for obs in obs_list:
        parsed = parse_fol_hypothesis(obs)
        if parsed:
            hyp_type, const, pred, negated = parsed
            if hyp_type == "ground":
                if not kb.can_derive(const, pred, negated):
                    return 0

    return 1


if __name__ == '__main__':
    main()
