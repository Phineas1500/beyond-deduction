"""
Difficulty levels and probability-based hiding logic.

Matches INABHYD's difficulty system exactly:
- SINGLE: Exactly one axiom hidden (simplest case)
- EASY: 10% probability of additional hiding
- MEDIUM: 20% probability of additional hiding
- HARD: 30% probability of additional hiding
"""
import enum
import random


class Difficulty(enum.Enum):
    """
    Difficulty levels matching INABHYD exactly.

    The value represents the probability of hiding additional axioms
    beyond the minimum required for the task.
    """
    SINGLE = 0.0    # Only one axiom hidden (simplest case)
    EASY = 0.1      # 10% chance of additional hiding
    MEDIUM = 0.2    # 20% chance of additional hiding
    HARD = 0.3      # 30% chance of additional hiding

    def should_hide(self) -> bool:
        """Probabilistically decide whether to hide an axiom."""
        return random.random() < self.value

    @property
    def missing_prob(self) -> float:
        """Return the missing probability value (for INABHYD compatibility)."""
        return self.value


class BiasedCoin:
    """Utility for probability-based decisions (matches INABHYD's util.py)."""

    @staticmethod
    def flip(prob: float) -> bool:
        """
        Return True with given probability.

        Args:
            prob: Probability of returning True (0.0 to 1.0)

        Returns:
            True with probability `prob`, False otherwise
        """
        if not 0 <= prob <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {prob}")
        return random.random() <= prob
