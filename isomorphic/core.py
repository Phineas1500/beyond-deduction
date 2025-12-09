"""
Core ontology structure with visibility tracking.

This is the unified representation from which both symbolic
and natural language renderings are derived. It matches INABHYD's
ontology.py implementation closely.

Key features:
- Visibility-tracked axioms (members, properties, parent-child relations)
- Three task types: INFER_PROPERTY, INFER_MEMBERSHIP, INFER_SUBTYPE
- Evidence allocation for hidden axioms
- Difficulty-based probabilistic hiding
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum, auto
import random

from .morphology import Morphology, Property
from .difficulty import Difficulty, BiasedCoin


class TaskType(Enum):
    """The three INABHYD task types."""
    INFER_PROPERTY = auto()     # recover_property: induce property rule
    INFER_MEMBERSHIP = auto()   # recover_membership: abduct entity type
    INFER_SUBTYPE = auto()      # recover_ontology: induce subtype relation


@dataclass
class OntologyConfig:
    """
    Configuration for ontology generation.

    Matches INABHYD's OntologyConfig exactly.
    """
    hops: int                           # Tree depth (number of reasoning hops)
    recover_membership: bool = False    # Enable INFER_MEMBERSHIP task
    recover_ontology: bool = False      # Enable INFER_SUBTYPE task
    recover_property: bool = False      # Enable INFER_PROPERTY task
    difficulty: Difficulty = Difficulty.EASY
    mix_hops: bool = False              # Allow hiding at any level (not just leaves/root)

    @property
    def task_types(self) -> List[TaskType]:
        """Get enabled task types."""
        types = []
        if self.recover_property:
            types.append(TaskType.INFER_PROPERTY)
        if self.recover_membership:
            types.append(TaskType.INFER_MEMBERSHIP)
        if self.recover_ontology:
            types.append(TaskType.INFER_SUBTYPE)
        return types

    @property
    def easiest_recover_membership(self) -> 'OntologyConfig':
        """Get easiest config for membership recovery only."""
        return OntologyConfig(self.hops, recover_membership=True)

    @property
    def easiest_recover_ontology(self) -> 'OntologyConfig':
        """Get easiest config for ontology recovery only."""
        return OntologyConfig(self.hops, recover_ontology=True)

    @property
    def easiest_recover_property(self) -> 'OntologyConfig':
        """Get easiest config for property recovery only."""
        return OntologyConfig(self.hops, recover_property=True)

    def __str__(self) -> str:
        parts = [f"{self.hops} hops"]
        if self.recover_membership:
            parts.append("membership")
        if self.recover_ontology:
            parts.append("ontology")
        if self.recover_property:
            parts.append("property")
        parts.append(self.difficulty.name.lower())
        if self.mix_hops:
            parts.append("mix")
        return "; ".join(parts)


class OntologyNode:
    """
    A concept node with visibility tracking for all axioms.

    Visibility allows hiding axioms to create reasoning tasks.
    This matches INABHYD's OntologyNode implementation.
    """

    def __init__(self, name: str, symbolic_id: int = -1, depth: int = 0):
        """
        Initialize an ontology node.

        Args:
            name: Natural language name (e.g., "wumpus")
            symbolic_id: Symbolic ID for rendering (c0, c1, etc.)
            depth: Depth in tree (0 = root)
        """
        self.name = name
        self.symbolic_id = symbolic_id
        self.depth = depth

        # Visibility-tracked collections: Dict[item, is_visible]
        self._members: Dict[str, bool] = {}
        self._properties: Dict[Property, bool] = {}
        self._parents: Dict['OntologyNode', bool] = {}
        self._children: Dict['OntologyNode', bool] = {}

        # Evidence tracking for task generation (matches INABHYD)
        self.associated_members_for_recover_ontology: List[Tuple['OntologyNode', str]] = []
        self.associated_members_for_recover_properties: Dict[Property, List[Tuple['OntologyNode', str]]] = defaultdict(list)
        self.associated_properties_for_recover_memberships: List[Tuple['OntologyNode', Property]] = []

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        if not isinstance(other, OntologyNode):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return f"OntologyNode({self.name}, id={self.symbolic_id}, depth={self.depth})"

    # ===== Member methods =====

    def add_member(self, member: str, visible: bool = True):
        """Add a member entity to this concept."""
        assert member not in self._members, f"Member {member} already exists"
        self._members[member] = visible

    def set_member_invisible(self, member: str):
        """Hide a member (for INFER_MEMBERSHIP tasks)."""
        assert member in self._members and self._members[member], f"Member {member} not found or already hidden"
        self._members[member] = False

    def members(self, visible: Optional[bool] = True) -> List[str]:
        """
        Get members filtered by visibility.

        Args:
            visible: True for visible only, False for hidden only, None for all
        """
        if visible is None:
            return list(self._members.keys())
        return [m for m, v in self._members.items() if v == visible]

    def num_members(self, visible: Optional[bool] = True) -> int:
        """Count members by visibility."""
        return len(self.members(visible))

    # ===== Property methods =====

    def add_property(self, prop: Property, visible: bool = True):
        """Add a property to this concept."""
        assert prop not in self._properties, f"Property {prop} already exists"
        assert prop.family not in self.prohibited_property_families, \
            f"Property family {prop.family} is prohibited for this node"
        self._properties[prop] = visible

    def set_property_invisible(self, prop: Property):
        """Hide a property (for INFER_PROPERTY tasks)."""
        assert prop in self._properties and self._properties[prop], \
            f"Property {prop} not found or already hidden"
        self._properties[prop] = False

    def properties(self, visible: Optional[bool] = True) -> List[Property]:
        """Get properties filtered by visibility."""
        if visible is None:
            return list(self._properties.keys())
        return [p for p, v in self._properties.items() if v == visible]

    def num_properties(self, visible: Optional[bool] = True) -> int:
        """Count properties by visibility."""
        return len(self.properties(visible))

    @property
    def property_families(self) -> Set[str]:
        """Get all property families used by this node."""
        return {p.family for p in self._properties.keys()}

    @property
    def prohibited_property_families(self) -> Set[str]:
        """
        Get families that cannot be used.

        A property family is prohibited if it's already used by this node,
        any ancestor, or any descendant.
        """
        families = set(self.property_families)

        # Check ancestors (BFS up)
        visited = set()
        queue = list(self.parents(None))
        while queue:
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)
            families.update(parent.property_families)
            queue.extend(parent.parents(None))

        # Check descendants (BFS down)
        visited = set()
        queue = list(self.children(None))
        while queue:
            child = queue.pop(0)
            if child in visited:
                continue
            visited.add(child)
            families.update(child.property_families)
            queue.extend(child.children(None))

        return families

    # ===== Parent methods =====

    def add_parent(self, parent: 'OntologyNode', visible: bool = True):
        """Add a parent (supertype) relation."""
        assert parent not in self._parents, f"Parent {parent} already exists"
        self._parents[parent] = visible

    def set_parent_invisible(self, parent: 'OntologyNode'):
        """Hide a parent relation (for INFER_SUBTYPE tasks)."""
        assert parent in self._parents and self._parents[parent], \
            f"Parent {parent} not found or already hidden"
        self._parents[parent] = False

    def parents(self, visible: Optional[bool] = True) -> List['OntologyNode']:
        """Get parents filtered by visibility."""
        if visible is None:
            return list(self._parents.keys())
        return [p for p, v in self._parents.items() if v == visible]

    def num_parents(self, visible: Optional[bool] = True) -> int:
        """Count parents by visibility."""
        return len(self.parents(visible))

    # ===== Child methods =====

    def add_child(self, child: 'OntologyNode', visible: bool = True):
        """Add a child (subtype) relation."""
        assert child not in self._children, f"Child {child} already exists"
        self._children[child] = visible

    def set_child_invisible(self, child: 'OntologyNode'):
        """Hide a child relation."""
        assert child in self._children and self._children[child], \
            f"Child {child} not found or already hidden"
        self._children[child] = False

    def children(self, visible: Optional[bool] = True) -> List['OntologyNode']:
        """Get children filtered by visibility."""
        if visible is None:
            return list(self._children.keys())
        return [c for c, v in self._children.items() if v == visible]

    def num_children(self, visible: Optional[bool] = True) -> int:
        """Count children by visibility."""
        return len(self.children(visible))


class Ontology:
    """
    The core ontology structure that generates isomorphic representations.

    This class builds the ontology tree and handles visibility-based
    axiom hiding for task generation. It provides the canonical structure
    that renderers transform into symbolic or natural language format.

    Matches INABHYD's Ontology class closely.
    """

    MIN_DISCRIMINANT = 2  # Minimum evidence pieces for valid inference
    MAX_CHILD_COUNT = MIN_DISCRIMINANT + 1  # = 3
    MIN_CHILD_COUNT = 1

    def __init__(self, config: OntologyConfig, seed: Optional[int] = None):
        """
        Initialize and build an ontology.

        Args:
            config: Configuration specifying task types and difficulty
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.config = config
        self.morphology = Morphology(seed)

        # Nodes organized by level
        self.nodes: List[List[OntologyNode]] = []
        self.pseudo_root: Optional[OntologyNode] = None
        self._next_symbolic_id = 0

        # Build the ontology
        self._build_ontology()
        self._take_missing_ontology()
        self._take_missing_property()
        self._allocate_members()
        self._take_missing_members()
        self._allocate_properties()

    @property
    def root(self) -> OntologyNode:
        """Get the root node."""
        return self.nodes[0][0]

    def _get_next_id(self) -> int:
        """Get next symbolic ID."""
        id = self._next_symbolic_id
        self._next_symbolic_id += 1
        return id

    def _build_ontology(self):
        """Build the basic tree structure (matches INABHYD's _build_ontology)."""
        # Create root
        root = OntologyNode(
            name=self.morphology.next_concept,
            symbolic_id=self._get_next_id(),
            depth=0
        )
        self.nodes = [[root]]

        # Add pseudo-root for subtype recovery (INFER_SUBTYPE)
        if self.config.recover_ontology:
            self.pseudo_root = OntologyNode(
                name=self.morphology.next_concept,
                symbolic_id=self._get_next_id(),
                depth=-1  # Above root
            )
            root.add_parent(self.pseudo_root)
            self.pseudo_root.add_child(root)

        # Build tree level by level
        for layer in range(self.config.hops - 1):
            self.nodes.append([])
            for parent in self.nodes[-2]:
                # Determine child count based on difficulty
                # (matches INABHYD's biased coin flip)
                num_children = self.MIN_CHILD_COUNT
                if layer == 0 or BiasedCoin.flip(1 - (1 - self.config.difficulty.missing_prob) ** 5):
                    num_children = self.MAX_CHILD_COUNT

                for _ in range(num_children):
                    child = OntologyNode(
                        name=self.morphology.next_concept,
                        symbolic_id=self._get_next_id(),
                        depth=layer + 1
                    )
                    child.add_parent(parent)
                    parent.add_child(child)
                    self.nodes[-1].append(child)

    def _take_missing_ontology(self):
        """Hide subtype relations for INFER_SUBTYPE tasks."""
        if not self.config.recover_ontology:
            return

        # Optionally hide relations at intermediate levels
        if self.config.mix_hops:
            for level_nodes in self.nodes:
                for node in level_nodes:
                    if (node.num_children(True) > self.MIN_DISCRIMINANT and
                            BiasedCoin.flip(self.config.difficulty.missing_prob)):
                        child = random.choice(node.children(True))
                        node.set_child_invisible(child)
                        child.set_parent_invisible(node)

        # Always hide root -> pseudo_root connection
        if self.pseudo_root:
            self.pseudo_root.set_child_invisible(self.root)
            self.root.set_parent_invisible(self.pseudo_root)

    def _take_missing_property(self):
        """Hide properties for INFER_PROPERTY tasks."""
        if not self.config.recover_property:
            return

        for level_nodes in self.nodes:
            for node in level_nodes:
                # Always hide on root, optionally on others based on mix_hops
                if (node == self.root or
                        (self.config.mix_hops and
                         BiasedCoin.flip(self.config.difficulty.missing_prob))):
                    try:
                        prop = self.morphology.next_property(node.prohibited_property_families)
                        node.add_property(prop, visible=True)  # Add first
                        node.set_property_invisible(prop)  # Then hide
                    except ValueError:
                        # No available properties
                        pass

    def _allocate_members(self):
        """
        Allocate entity members as evidence for hidden axioms.
        Matches INABHYD's _allocate_members method.
        """
        for level_nodes in self.nodes:
            for node in level_nodes:
                # Evidence for hidden subtype relations (INFER_SUBTYPE)
                if node.num_parents(False) > 0:  # Has hidden parent
                    assert node.num_parents(False) <= 1
                    # Add a direct member
                    member = self.morphology.next_entity
                    node.add_member(member)
                    node.associated_members_for_recover_ontology.append((node, member))

                    # Get candidate descendant nodes
                    candidate_nodes = self._get_descendant_levels(node)

                    # Add MIN_DISCRIMINANT more members from descendants
                    for _ in range(self.MIN_DISCRIMINANT):
                        if not self.config.mix_hops:
                            # Only from deepest level
                            chosen_node = random.choice(candidate_nodes[-1]) if candidate_nodes else node
                        else:
                            # From any level
                            all_descendants = [n for level in candidate_nodes for n in level]
                            chosen_node = random.choice(all_descendants) if all_descendants else node

                        member = self.morphology.next_entity
                        chosen_node.add_member(member)
                        node.associated_members_for_recover_ontology.append((chosen_node, member))

                # Evidence for hidden properties (INFER_PROPERTY)
                if node.num_properties(False) > 0:  # Has hidden property
                    assert node.num_properties(False) <= 1
                    prop = node.properties(False)[0]

                    # Add a direct member
                    member = self.morphology.next_entity
                    node.add_member(member)
                    node.associated_members_for_recover_properties[prop].append((node, member))

                    # Get candidate descendant nodes
                    candidate_nodes = self._get_descendant_levels(node)

                    # Add MIN_DISCRIMINANT more members from descendants
                    for _ in range(self.MIN_DISCRIMINANT):
                        if not self.config.mix_hops:
                            chosen_node = random.choice(candidate_nodes[-1]) if candidate_nodes else node
                        else:
                            all_descendants = [n for level in candidate_nodes for n in level]
                            chosen_node = random.choice(all_descendants) if all_descendants else node

                        member = self.morphology.next_entity
                        chosen_node.add_member(member)
                        node.associated_members_for_recover_properties[prop].append((chosen_node, member))

    def _get_descendant_levels(self, node: OntologyNode) -> List[List[OntologyNode]]:
        """Get descendants organized by level."""
        levels = [[node]]
        while True:
            next_level = []
            for n in levels[-1]:
                next_level.extend(n.children(None))
            if not next_level:
                break
            levels.append(next_level)
        return levels[1:]  # Exclude the starting node

    def _take_missing_members(self):
        """Hide memberships for INFER_MEMBERSHIP tasks."""
        if not self.config.recover_membership:
            return

        # Ensure all nodes have at least one member
        for level_nodes in self.nodes:
            for node in level_nodes:
                if node.num_members(None) == 0:
                    node.add_member(self.morphology.next_entity)

        # Find leaf nodes
        leaf_nodes = self.nodes[-1] if self.nodes else []

        if self.config.difficulty == Difficulty.SINGLE:
            # Hide exactly one membership at a leaf
            chosen_node = random.choice(leaf_nodes)
            hidden_member = random.choice(chosen_node.members(True))
            chosen_node.set_member_invisible(hidden_member)
            return

        # Probabilistic hiding based on difficulty
        for level_nodes in self.nodes:
            for node in level_nodes:
                if node in leaf_nodes or self.config.mix_hops:
                    if (BiasedCoin.flip(self.config.difficulty.missing_prob) and
                            node.num_members(True) > 0):
                        hidden_member = random.choice(node.members(True))
                        node.set_member_invisible(hidden_member)

        # Ensure at least one membership is hidden at a leaf
        has_hidden = any(node.num_members(False) > 0 for node in leaf_nodes)
        if not has_hidden:
            chosen_node = random.choice(leaf_nodes)
            if chosen_node.num_members(True) > 0:
                hidden_member = random.choice(chosen_node.members(True))
                chosen_node.set_member_invisible(hidden_member)

    def _allocate_properties(self):
        """
        Allocate properties as evidence for hidden memberships.
        Matches INABHYD's _allocate_properties method.
        """
        for level_nodes in self.nodes:
            for node in level_nodes:
                if node.num_members(False) > 0:  # Has hidden membership
                    # Add a property to this node
                    try:
                        prop = self.morphology.next_property(node.prohibited_property_families)
                        node.add_property(prop)
                        node.associated_properties_for_recover_memberships.append((node, prop))
                    except ValueError:
                        pass

                    # Get ancestor nodes
                    ancestors = self._get_ancestors(node)

                    # Add MIN_DISCRIMINANT more properties from ancestors
                    for _ in range(self.MIN_DISCRIMINANT):
                        if not self.config.mix_hops:
                            chosen_node = ancestors[-1] if ancestors else node
                        else:
                            chosen_node = random.choice(ancestors) if ancestors else node

                        # Try to reuse existing property
                        existing_props = chosen_node.properties(True)
                        prop_used = [p for _, p in node.associated_properties_for_recover_memberships]

                        reusable = [p for p in existing_props if p not in prop_used]
                        if reusable and len(node.associated_properties_for_recover_memberships) < self.MIN_DISCRIMINANT + 1:
                            prop = reusable[0]
                            node.associated_properties_for_recover_memberships.append((chosen_node, prop))
                        else:
                            # Add new property
                            try:
                                prop = self.morphology.next_property(chosen_node.prohibited_property_families)
                                chosen_node.add_property(prop)
                                node.associated_properties_for_recover_memberships.append((chosen_node, prop))
                            except ValueError:
                                pass

    def _get_ancestors(self, node: OntologyNode) -> List[OntologyNode]:
        """Get all ancestors (root first)."""
        ancestors = []
        current = node
        while current.parents(None):
            parent = current.parents(None)[0]
            ancestors.insert(0, parent)
            current = parent
        return ancestors

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Get all hidden axioms as hypotheses (the answers to infer).

        Returns:
            List of hypothesis dicts with task_type, subject, predicate, depth
        """
        hypotheses = []

        for level_nodes in self.nodes:
            for node in level_nodes:
                # Hidden subtype relations
                for parent in node.parents(False):
                    hypotheses.append({
                        "task_type": TaskType.INFER_SUBTYPE,
                        "subject": node,
                        "predicate": parent,
                        "depth": node.depth
                    })

                # Hidden properties
                for prop in node.properties(False):
                    hypotheses.append({
                        "task_type": TaskType.INFER_PROPERTY,
                        "subject": node,
                        "predicate": prop,
                        "depth": node.depth
                    })

                # Hidden memberships
                for member in node.members(False):
                    hypotheses.append({
                        "task_type": TaskType.INFER_MEMBERSHIP,
                        "subject": member,
                        "predicate": node,
                        "depth": node.depth
                    })

        return hypotheses

    def get_observations(self) -> List[Dict[str, Any]]:
        """
        Get all observations (evidence for hidden axioms).

        Returns:
            List of observation dicts with subject (entity), predicate, source_node
        """
        observations = []

        for level_nodes in self.nodes:
            for node in level_nodes:
                # Observations for hidden subtypes
                if node.parents(False):
                    parent = node.parents(False)[0]
                    for (src_node, member) in node.associated_members_for_recover_ontology:
                        observations.append({
                            "subject": member,
                            "predicate": parent,  # Entity belongs to hidden parent concept
                            "source_node": src_node
                        })

                # Observations for hidden properties
                for prop in node.properties(False):
                    for (src_node, member) in node.associated_members_for_recover_properties[prop]:
                        observations.append({
                            "subject": member,
                            "predicate": prop,  # Entity has hidden property
                            "source_node": src_node
                        })

                # Observations for hidden memberships
                for member in node.members(False):
                    for (src_node, prop) in node.associated_properties_for_recover_memberships:
                        observations.append({
                            "subject": member,
                            "predicate": prop,  # Hidden entity has property
                            "source_node": src_node
                        })

        return observations

    def get_visible_theories(self) -> List[Dict[str, Any]]:
        """
        Get all visible (non-hidden) axioms as the world model/theories.

        Returns:
            List of theory dicts with axiom_type, subject, predicate
        """
        theories = []

        for level_nodes in self.nodes:
            for node in level_nodes:
                # Visible subtype relations
                for parent in node.parents(True):
                    theories.append({
                        "axiom_type": "subtype",
                        "subject": node,
                        "predicate": parent
                    })

                # Visible properties
                for prop in node.properties(True):
                    theories.append({
                        "axiom_type": "property",
                        "subject": node,
                        "predicate": prop
                    })

                # Visible memberships
                for member in node.members(True):
                    theories.append({
                        "axiom_type": "membership",
                        "subject": member,
                        "predicate": node
                    })

        return theories

    def get_chain_of_thought(self) -> List[Dict[str, Any]]:
        """
        Generate chain-of-thought reasoning steps.

        This matches INABHYD's CoT property, generating step-by-step
        reasoning that explains how to derive each hidden hypothesis
        from the observations.

        Returns:
            List of reasoning step dicts with:
            - step_type: "chain", "property", "suppose", "conclude"
            - subject: entity or concept
            - predicate: concept or property
            - text: human-readable description
        """
        cot_steps = []

        for level_nodes in self.nodes:
            for node in level_nodes:
                # CoT for hidden memberships (INFER_MEMBERSHIP)
                if node.members(False):
                    for hidden_member in node.members(False):
                        for (chosen_node, prop) in node.associated_properties_for_recover_memberships:
                            # Build chain from node to chosen_node (going up the tree)
                            if chosen_node != node:
                                chains = [node]
                                current = node
                                while current != chosen_node and current.parents(True):
                                    parent = current.parents(True)[0]
                                    chains.append(parent)
                                    current = parent

                                # Add chain steps (concept -> parent concept)
                                for i in range(1, len(chains)):
                                    # "X is a Y" (subtype relation)
                                    cot_steps.append({
                                        "step_type": "chain",
                                        "subject": chains[i - 1],
                                        "predicate": chains[i],
                                        "text": f"{chains[i-1].name} is a subtype of {chains[i].name}"
                                    })
                                    # "Entity is a Y" (derived membership)
                                    cot_steps.append({
                                        "step_type": "derive_membership",
                                        "subject": hidden_member,
                                        "predicate": chains[i],
                                        "text": f"{hidden_member} is a {chains[i].name}"
                                    })

                            # "Y has property P" (property of ancestor)
                            cot_steps.append({
                                "step_type": "property",
                                "subject": chosen_node,
                                "predicate": prop,
                                "text": f"Every {chosen_node.name} is {prop}"
                            })

                            # "Suppose entity is a X" (the hypothesis)
                            cot_steps.append({
                                "step_type": "suppose",
                                "subject": hidden_member,
                                "predicate": node,
                                "text": f"Suppose {hidden_member} is a {node.name}"
                            })

                            # "Therefore entity has property P" (conclusion)
                            cot_steps.append({
                                "step_type": "conclude",
                                "subject": hidden_member,
                                "predicate": prop,
                                "text": f"{hidden_member} is {prop}"
                            })

                # CoT for hidden properties (INFER_PROPERTY)
                for prop in node.properties(False):
                    for (chosen_node, member) in node.associated_members_for_recover_properties[prop]:
                        # Build chain from chosen_node up to node
                        if chosen_node != node:
                            current = chosen_node
                            while current != node and current.parents(True):
                                parent = current.parents(True)[0]
                                # "Entity is a current_concept"
                                cot_steps.append({
                                    "step_type": "membership",
                                    "subject": member,
                                    "predicate": current,
                                    "text": f"{member} is a {current.name}"
                                })
                                # "current_concept is a parent_concept"
                                cot_steps.append({
                                    "step_type": "chain",
                                    "subject": current,
                                    "predicate": parent,
                                    "text": f"{current.name} is a subtype of {parent.name}"
                                })
                                current = parent

                        # "Entity is a node" (final membership in target concept)
                        cot_steps.append({
                            "step_type": "membership",
                            "subject": member,
                            "predicate": node,
                            "text": f"{member} is a {node.name}"
                        })

                        # "Suppose node has property P" (the hypothesis)
                        cot_steps.append({
                            "step_type": "suppose",
                            "subject": node,
                            "predicate": prop,
                            "text": f"Suppose every {node.name} is {prop}"
                        })

                        # "Therefore entity has P" (conclusion)
                        cot_steps.append({
                            "step_type": "conclude",
                            "subject": member,
                            "predicate": prop,
                            "text": f"{member} is {prop}"
                        })

                # CoT for hidden subtype relations (INFER_SUBTYPE)
                if node.parents(False):
                    hidden_parent = node.parents(False)[0]
                    for (chosen_node, member) in node.associated_members_for_recover_ontology:
                        # Build chain from chosen_node up to node
                        if chosen_node != node:
                            current = chosen_node
                            while current != node and current.parents(True):
                                parent = current.parents(True)[0]
                                # "Entity is a current_concept"
                                cot_steps.append({
                                    "step_type": "membership",
                                    "subject": member,
                                    "predicate": current,
                                    "text": f"{member} is a {current.name}"
                                })
                                # "current_concept is a parent_concept"
                                cot_steps.append({
                                    "step_type": "chain",
                                    "subject": current,
                                    "predicate": parent,
                                    "text": f"{current.name} is a subtype of {parent.name}"
                                })
                                current = parent

                        # "Entity is a node"
                        cot_steps.append({
                            "step_type": "membership",
                            "subject": member,
                            "predicate": node,
                            "text": f"{member} is a {node.name}"
                        })

                        # "Suppose node is a hidden_parent" (the hypothesis)
                        cot_steps.append({
                            "step_type": "suppose",
                            "subject": node,
                            "predicate": hidden_parent,
                            "text": f"Suppose every {node.name} is a {hidden_parent.name}"
                        })

                        # "Therefore entity is a hidden_parent" (conclusion)
                        cot_steps.append({
                            "step_type": "conclude",
                            "subject": member,
                            "predicate": hidden_parent,
                            "text": f"{member} is a {hidden_parent.name}"
                        })

        return cot_steps

    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all nodes in the ontology (including pseudo_root if exists)."""
        all_nodes = [n for level in self.nodes for n in level]
        if self.pseudo_root:
            all_nodes.append(self.pseudo_root)
        return all_nodes

    def print_ontology(self):
        """Print the ontology tree for debugging."""
        if self.pseudo_root:
            self._print_node(self.pseudo_root, 0)
        else:
            self._print_node(self.root, 0)

    def _print_node(self, node: OntologyNode, indent: int):
        """Recursively print a node and its children."""
        prefix = "  " * indent
        props_str = ", ".join(str(p) for p in node.properties(None))
        members_str = ", ".join(node.members(None))
        hidden_suffix = " [HIDDEN]" if node.parents(False) else ""

        print(f"{prefix}{node.name} (c{node.symbolic_id}){hidden_suffix}")
        if props_str:
            print(f"{prefix}  props: {props_str}")
        if members_str:
            print(f"{prefix}  members: {members_str}")

        for child in node.children(True):
            self._print_node(child, indent + 1)
        for child in node.children(False):
            print(f"{prefix}  [HIDDEN CHILD]")
            self._print_node(child, indent + 1)
