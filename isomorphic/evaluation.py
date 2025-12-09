"""
Parsimony evaluation metrics for INABHYD tasks.

Implements:
- Weak accuracy: Any valid hypothesis (explains observations)
- Strong accuracy: Most parsimonious hypothesis (exact match to ground truth)
- Quality score: INABHYD's parsimony-based quality metric

The quality score formula from INABHYD:
    q(H) = (1/|H| × Σ n(h)) / (1/|H*| × Σ n(h*))

Where:
- H = predicted hypotheses
- H* = ground truth (most parsimonious) hypotheses
- n(h) = number of observations explained by hypothesis h

A more general hypothesis explains MORE observations, so gets a higher n(h).
"""
from typing import List, Dict, Any, Set, Optional, Tuple
from .core import Ontology, OntologyNode, TaskType
from .morphology import Property


def normalize_hypothesis(hyp: str) -> str:
    """
    Normalize hypothesis for comparison.

    Removes whitespace variations and lowercases.

    Args:
        hyp: Hypothesis string

    Returns:
        Normalized string
    """
    return hyp.lower().strip().replace(" ", "")


def extract_hypothesis_components(hyp: str) -> Dict[str, Any]:
    """
    Extract components from a hypothesis string.

    Args:
        hyp: Hypothesis in symbolic FOL or natural language

    Returns:
        Dict with subject, predicate, hypothesis_type
    """
    hyp = hyp.strip()

    # Try to parse symbolic format: "forall x: cN(x) -> pM(x)" or "cN(eM)"
    if "forall" in hyp.lower():
        # Quantified rule
        import re
        match = re.search(r'([c~p]\d+)\(x\)\s*->\s*([c~p]\d+)\(x\)', hyp)
        if match:
            return {
                "subject": match.group(1),
                "predicate": match.group(2),
                "type": "rule"
            }
    else:
        # Ground fact
        import re
        match = re.search(r'([cp]\d+)\(([e]\d+)\)', hyp)
        if match:
            return {
                "subject": match.group(2),
                "predicate": match.group(1),
                "type": "fact"
            }

    return {"subject": None, "predicate": None, "type": "unknown"}


def compute_parsimony_score(prediction: str, ground_truth: str,
                            alternatives: Optional[List[str]] = None) -> float:
    """
    Compute simplified parsimony score for a prediction.

    This is a fallback when ontology structure is not available.
    For the full INABHYD quality score, use QualityScoreCalculator.

    Args:
        prediction: The predicted hypothesis
        ground_truth: The most parsimonious hypothesis (score 1.0)
        alternatives: Less parsimonious valid alternatives

    Returns:
        Score from 0.0 (invalid) to 1.0 (most parsimonious)
    """
    pred_norm = normalize_hypothesis(prediction)
    gt_norm = normalize_hypothesis(ground_truth)

    # Perfect match
    if pred_norm == gt_norm:
        return 1.0

    # Check if it's a valid alternative
    if alternatives:
        for i, alt in enumerate(alternatives):
            if normalize_hypothesis(alt) == pred_norm:
                # Score decreases with specificity (later alternatives are less parsimonious)
                return 0.5 * (1 - i / len(alternatives))

    return 0.0


# =============================================================================
# INABHYD Quality Score Implementation
# =============================================================================

class QualityScoreCalculator:
    """
    Computes the full INABHYD quality score based on observation coverage.

    The quality score measures how parsimonious a hypothesis is:
        q(H) = (1/|H| × Σ n(h)) / (1/|H*| × Σ n(h*))

    Where n(h) = number of observations explained by hypothesis h.

    A more general hypothesis (e.g., "All mammals are cute") explains more
    observations than a specific one (e.g., "All cats are cute"), giving
    it a higher n(h) value.
    """

    def __init__(self, ontology: Ontology):
        """
        Initialize with an ontology structure.

        Args:
            ontology: The Ontology instance containing the full structure
        """
        self.ontology = ontology
        self._build_indexes()

    def _build_indexes(self):
        """Build indexes for efficient observation counting."""
        # Map concept names to nodes
        self.concept_to_node: Dict[str, OntologyNode] = {}
        for node in self.ontology.get_all_nodes():
            self.concept_to_node[node.name] = node

        # Map entity names to their concept membership
        self.entity_to_concepts: Dict[str, Set[str]] = {}
        for node in self.ontology.get_all_nodes():
            for member in node.members(None):  # All members (visible + hidden)
                if member not in self.entity_to_concepts:
                    self.entity_to_concepts[member] = set()
                # Add this concept and all ancestors
                self.entity_to_concepts[member].add(node.name)
                for ancestor in self._get_all_ancestors(node):
                    self.entity_to_concepts[member].add(ancestor.name)

        # Cache observations
        self.observations = self.ontology.get_observations()

    def _get_all_ancestors(self, node: OntologyNode) -> List[OntologyNode]:
        """Get all ancestor nodes (including through hidden relations)."""
        ancestors = []
        visited = set()
        queue = list(node.parents(None))  # Include hidden parents
        while queue:
            parent = queue.pop(0)
            if parent.name in visited:
                continue
            visited.add(parent.name)
            ancestors.append(parent)
            queue.extend(parent.parents(None))
        return ancestors

    def _get_all_descendants(self, node: OntologyNode) -> List[OntologyNode]:
        """Get all descendant nodes (including through hidden relations)."""
        descendants = []
        visited = set()
        queue = list(node.children(None))  # Include hidden children
        while queue:
            child = queue.pop(0)
            if child.name in visited:
                continue
            visited.add(child.name)
            descendants.append(child)
            queue.extend(child.children(None))
        return descendants

    def count_observations_explained(self, hypothesis: Dict[str, Any]) -> int:
        """
        Count how many observations are explained by a hypothesis.

        Args:
            hypothesis: Dict with task_type, subject, predicate

        Returns:
            Number of observations explained by this hypothesis
        """
        task_type = hypothesis["task_type"]
        count = 0

        if task_type == TaskType.INFER_PROPERTY:
            # Hypothesis: "All X have property P"
            # Explains: All observations of entities with P that are members of X or its subtypes
            concept_node = hypothesis["subject"]
            prop = hypothesis["predicate"]

            # Get all entities that are members of this concept or its descendants
            covered_entities = set()
            covered_entities.update(concept_node.members(None))
            for desc in self._get_all_descendants(concept_node):
                covered_entities.update(desc.members(None))

            # Count observations where entity has this property
            for obs in self.observations:
                if (isinstance(obs["predicate"], Property) and
                    obs["predicate"] == prop and
                    obs["subject"] in covered_entities):
                    count += 1

        elif task_type == TaskType.INFER_MEMBERSHIP:
            # Hypothesis: "Entity E is a member of concept C"
            # Explains: All observations of properties that E has via C
            entity = hypothesis["subject"]
            concept_node = hypothesis["predicate"]

            # Get all properties this entity would have through this concept
            # (properties of concept and all ancestors)
            covered_props = set()
            covered_props.update(concept_node.properties(None))
            for ancestor in self._get_all_ancestors(concept_node):
                covered_props.update(ancestor.properties(None))

            # Count observations where this entity has these properties
            for obs in self.observations:
                if (obs["subject"] == entity and
                    isinstance(obs["predicate"], Property) and
                    obs["predicate"] in covered_props):
                    count += 1

        elif task_type == TaskType.INFER_SUBTYPE:
            # Hypothesis: "All X are Y" (X is subtype of Y)
            # Explains: All observations showing members of X have Y's properties
            child_node = hypothesis["subject"]
            parent_node = hypothesis["predicate"]

            # Get all entities that are members of child or its descendants
            covered_entities = set()
            covered_entities.update(child_node.members(None))
            for desc in self._get_all_descendants(child_node):
                covered_entities.update(desc.members(None))

            # Count observations where these entities are shown to be in parent concept
            for obs in self.observations:
                if (isinstance(obs["predicate"], OntologyNode) and
                    obs["predicate"] == parent_node and
                    obs["subject"] in covered_entities):
                    count += 1

        return max(count, 1)  # At least 1 to avoid division issues

    def compute_quality_score(
        self,
        predicted_hypotheses: List[Dict[str, Any]],
        ground_truth_hypotheses: List[Dict[str, Any]]
    ) -> float:
        """
        Compute the INABHYD quality score.

        q(H) = (1/|H| × Σ n(h)) / (1/|H*| × Σ n(h*))

        Args:
            predicted_hypotheses: List of predicted hypothesis dicts
            ground_truth_hypotheses: List of ground truth hypothesis dicts

        Returns:
            Quality score (0.0 to 1.0+, where 1.0 = matches ground truth quality)
        """
        if not predicted_hypotheses or not ground_truth_hypotheses:
            return 0.0

        # Compute numerator: average n(h) for predictions
        pred_total = sum(self.count_observations_explained(h) for h in predicted_hypotheses)
        pred_avg = pred_total / len(predicted_hypotheses)

        # Compute denominator: average n(h*) for ground truth
        gt_total = sum(self.count_observations_explained(h) for h in ground_truth_hypotheses)
        gt_avg = gt_total / len(ground_truth_hypotheses)

        if gt_avg == 0:
            return 0.0

        return pred_avg / gt_avg

    def generate_alternative_hypotheses(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate valid but less parsimonious alternative hypotheses.

        For a ground truth hypothesis at concept C, generate alternatives
        for all child concepts (which are less general).

        Args:
            ground_truth: The ground truth hypothesis dict

        Returns:
            List of alternative hypothesis dicts, ordered by decreasing generality
        """
        alternatives = []
        task_type = ground_truth["task_type"]

        if task_type == TaskType.INFER_PROPERTY:
            # For "All X have P", alternatives are "All X_child have P"
            concept_node = ground_truth["subject"]
            prop = ground_truth["predicate"]

            # Get all descendants (less general concepts)
            for desc in self._get_all_descendants(concept_node):
                alternatives.append({
                    "task_type": TaskType.INFER_PROPERTY,
                    "subject": desc,
                    "predicate": prop,
                    "depth": desc.depth
                })

        elif task_type == TaskType.INFER_SUBTYPE:
            # For "All X are Y", alternatives are "All X_child are Y"
            child_node = ground_truth["subject"]
            parent_node = ground_truth["predicate"]

            for desc in self._get_all_descendants(child_node):
                alternatives.append({
                    "task_type": TaskType.INFER_SUBTYPE,
                    "subject": desc,
                    "predicate": parent_node,
                    "depth": desc.depth
                })

        # Sort by depth (shallower = more general = higher priority)
        alternatives.sort(key=lambda x: x.get("depth", 0))

        return alternatives


def compute_quality_score_from_ontology(
    ontology: Ontology,
    predicted_hypotheses: List[Dict[str, Any]],
    ground_truth_hypotheses: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Convenience function to compute quality score from an ontology.

    Args:
        ontology: The Ontology instance
        predicted_hypotheses: List of predicted hypothesis dicts
        ground_truth_hypotheses: Ground truth (defaults to ontology.get_hypotheses())

    Returns:
        Quality score (0.0 to 1.0+)
    """
    if ground_truth_hypotheses is None:
        ground_truth_hypotheses = ontology.get_hypotheses()

    calculator = QualityScoreCalculator(ontology)
    return calculator.compute_quality_score(predicted_hypotheses, ground_truth_hypotheses)


def weak_accuracy(predictions: List[str], ground_truths: List[str],
                  alternatives: Optional[List[List[str]]] = None) -> float:
    """
    Compute weak accuracy: proportion of valid hypotheses.

    A prediction is correct if it matches the ground truth OR any valid alternative.

    Args:
        predictions: List of predicted hypotheses
        ground_truths: List of ground truth hypotheses
        alternatives: List of valid alternative hypotheses per sample

    Returns:
        Proportion correct (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    correct = 0
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_norm = normalize_hypothesis(pred)
        gt_norm = normalize_hypothesis(gt)

        # Check ground truth
        if pred_norm == gt_norm:
            correct += 1
            continue

        # Check alternatives
        if alternatives and i < len(alternatives):
            for alt in alternatives[i]:
                if normalize_hypothesis(alt) == pred_norm:
                    correct += 1
                    break

    return correct / len(predictions)


def strong_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute strong accuracy: proportion matching ground truth exactly.

    Only the most parsimonious hypothesis is considered correct.

    Args:
        predictions: List of predicted hypotheses
        ground_truths: List of ground truth hypotheses

    Returns:
        Proportion exactly matching (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truths)
                  if normalize_hypothesis(p) == normalize_hypothesis(g))
    return correct / len(predictions)


class ParsimonyEvaluator:
    """
    Evaluates model predictions against INABHYD criteria.

    Tracks results across multiple samples and computes aggregate metrics.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results: List[Dict[str, Any]] = []

    def reset(self):
        """Reset all tracked results."""
        self.results = []

    def evaluate_sample(
        self,
        prediction: str,
        ground_truth: str,
        alternatives: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Args:
            prediction: The model's predicted hypothesis
            ground_truth: The correct most-parsimonious hypothesis
            alternatives: Valid but less parsimonious alternatives
            metadata: Additional metadata to store

        Returns:
            Dict with evaluation results
        """
        pred_norm = normalize_hypothesis(prediction)
        gt_norm = normalize_hypothesis(ground_truth)

        # Check exact match
        exact_match = pred_norm == gt_norm

        # Check if valid (matches ground truth or any alternative)
        is_valid = exact_match
        matched_alternative = None
        if not is_valid and alternatives:
            for i, alt in enumerate(alternatives):
                if normalize_hypothesis(alt) == pred_norm:
                    is_valid = True
                    matched_alternative = i
                    break

        # Compute parsimony score
        score = compute_parsimony_score(prediction, ground_truth, alternatives)

        result = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "is_valid": is_valid,
            "matched_alternative": matched_alternative,
            "parsimony_score": score,
            "metadata": metadata or {}
        }

        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        alternatives: Optional[List[List[str]]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: List of predicted hypotheses
            ground_truths: List of ground truth hypotheses
            alternatives: List of valid alternatives per sample
            metadata_list: List of metadata dicts per sample

        Returns:
            List of evaluation result dicts
        """
        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            alts = alternatives[i] if alternatives and i < len(alternatives) else None
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            result = self.evaluate_sample(pred, gt, alts, meta)
            results.append(result)
        return results

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all evaluated samples.

        Returns:
            Dict with strong_accuracy, weak_accuracy, mean_parsimony, total_samples
        """
        if not self.results:
            return {
                "strong_accuracy": 0.0,
                "weak_accuracy": 0.0,
                "mean_parsimony": 0.0,
                "total_samples": 0
            }

        n = len(self.results)

        return {
            "strong_accuracy": sum(r["exact_match"] for r in self.results) / n,
            "weak_accuracy": sum(r["is_valid"] for r in self.results) / n,
            "mean_parsimony": sum(r["parsimony_score"] for r in self.results) / n,
            "total_samples": n
        }

    def compute_metrics_by_task_type(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute metrics grouped by task type.

        Returns:
            Dict mapping task type to metrics dict
        """
        by_type: Dict[str, List[Dict]] = {}

        for r in self.results:
            task_type = r.get("metadata", {}).get("task_type", "unknown")
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(r)

        metrics = {}
        for task_type, results in by_type.items():
            n = len(results)
            metrics[task_type] = {
                "strong_accuracy": sum(r["exact_match"] for r in results) / n,
                "weak_accuracy": sum(r["is_valid"] for r in results) / n,
                "mean_parsimony": sum(r["parsimony_score"] for r in results) / n,
                "total_samples": n
            }

        return metrics

    def compute_metrics_by_depth(self) -> Dict[int, Dict[str, Any]]:
        """
        Compute metrics grouped by reasoning depth.

        Returns:
            Dict mapping depth to metrics dict
        """
        by_depth: Dict[int, List[Dict]] = {}

        for r in self.results:
            depth = r.get("metadata", {}).get("depth", -1)
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(r)

        metrics = {}
        for depth, results in by_depth.items():
            n = len(results)
            metrics[depth] = {
                "strong_accuracy": sum(r["exact_match"] for r in results) / n,
                "weak_accuracy": sum(r["is_valid"] for r in results) / n,
                "mean_parsimony": sum(r["parsimony_score"] for r in results) / n,
                "total_samples": n
            }

        return metrics

    def get_failures(self) -> List[Dict[str, Any]]:
        """Get all samples where the prediction was incorrect."""
        return [r for r in self.results if not r["exact_match"]]

    def get_summary(self) -> str:
        """Get a human-readable summary of results."""
        metrics = self.compute_metrics()
        lines = [
            f"Total samples: {metrics['total_samples']}",
            f"Strong accuracy: {metrics['strong_accuracy']:.1%}",
            f"Weak accuracy: {metrics['weak_accuracy']:.1%}",
            f"Mean parsimony: {metrics['mean_parsimony']:.3f}",
        ]
        return "\n".join(lines)
