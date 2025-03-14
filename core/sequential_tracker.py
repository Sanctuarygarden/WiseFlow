import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import numpy as np
from collections import defaultdict
import logging

# Import LangChain components
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContextTracker")

# ============= Concept Relationship Modeling =============

@dataclass
class ConceptRelationship:
    """
    Represents a relationship between two concepts in the conversation knowledge graph
    
    This structure captures:
    1. Source and target concepts
    2. Type of relationship (depends_on, refines, etc.)
    3. Strength and directionality of relationship
    4. Timestamp and entry where relationship was established
    """
    source_concept: str
    target_concept: str
    relationship_type: str  # e.g., "depends_on", "refines", "conflicts_with", etc.
    strength: float  # 0.0-1.0
    bidirectional: bool = False
    description: str = ""
    first_mentioned: str = field(default_factory=lambda: datetime.now().isoformat())
    entry_id: Optional[str] = None
    confidence: float = 0.8  # Confidence in this relationship existing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptRelationship':
        """Create instance from dictionary"""
        return cls(**data)
    
    def __hash__(self):
        """Hash function for using in sets"""
        if self.bidirectional:
            # For bidirectional relationships, order doesn't matter
            concepts = sorted([self.source_concept, self.target_concept])
            return hash((concepts[0], concepts[1], self.relationship_type))
        else:
            return hash((self.source_concept, self.target_concept, self.relationship_type))
    
    def __eq__(self, other):
        """Equality check for using in sets"""
        if not isinstance(other, ConceptRelationship):
            return False
            
        if self.bidirectional:
            # For bidirectional relationships, check both directions
            return (
                ((self.source_concept == other.source_concept and 
                  self.target_concept == other.target_concept) or
                 (self.source_concept == other.target_concept and
                  self.target_concept == other.source_concept)) and
                self.relationship_type == other.relationship_type
            )
        else:
            return (
                self.source_concept == other.source_concept and
                self.target_concept == other.target_concept and
                self.relationship_type == other.relationship_type
            )

# ============= Priority-Based Context Management =============

class PriorityTier:
    """
    Priority tiers for concept and theme importance
    """
    CORE = "core"  # Core project vision, fundamental concepts (highest priority)
    ACTIVE = "active"  # Currently active development concepts
    SUPPORTING = "supporting"  # Supporting ideas and context
    BACKGROUND = "background"  # Background information (lowest priority)
    
    @staticmethod
    def get_importance_score(tier: str) -> float:
        """Convert priority tier to importance score (0.0-1.0)"""
        tiers = {
            PriorityTier.CORE: 0.9,
            PriorityTier.ACTIVE: 0.7,
            PriorityTier.SUPPORTING: 0.5,
            PriorityTier.BACKGROUND: 0.3
        }
        return tiers.get(tier, 0.5)
    
    @staticmethod
    def get_tier_from_score(score: float) -> str:
        """Determine priority tier from importance score"""
        if score >= 0.8:
            return PriorityTier.CORE
        elif score >= 0.6:
            return PriorityTier.ACTIVE
        elif score >= 0.4:
            return PriorityTier.SUPPORTING
        else:
            return PriorityTier.BACKGROUND

# ============= Enhanced Data Structures =============

@dataclass
class ConversationEntry:
    """Enhanced conversation turn with hierarchical context tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_input: str = ""
    system_response: str = ""
    
    # Parent-child relationships for logical tracking
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # Hierarchical metadata with importance levels
    themes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {theme: {importance: score, first_mentioned: timestamp}}
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {concept: {importance: score, definition: str, evolution: []}}
    contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {context: {importance: score, status: active/deprecated}}
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    evolution_markers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Vector embeddings
    embedding: Optional[List[float]] = None
    
    # Memory management metadata
    memory_tier: str = "ram"  # Options: "ram", "local", "nas"
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Logic validation markers
    validated: bool = False
    validation_notes: Optional[str] = None
    
    # New fields for enhanced tracking
    priority_tier: str = PriorityTier.ACTIVE  # Default to ACTIVE tier
    concept_confidences: Dict[str, float] = field(default_factory=dict)  # concept -> confidence
    theme_confidences: Dict[str, float] = field(default_factory=dict)  # theme -> confidence
    concept_relationships: List[ConceptRelationship] = field(default_factory=list)  # Relationships established in this entry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {k: v for k, v in self.__dict__.items() if k != 'embedding' and k != 'concept_relationships'}
        
        # Handle embedding
        if self.embedding is not None:
            result['embedding'] = list(self.embedding)
        
        # Handle concept relationships
        result['concept_relationships'] = [rel.to_dict() for rel in self.concept_relationships]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create instance from dictionary"""
        # Handle embedding conversion if present
        embedding = data.pop('embedding', None)
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float32)
        
        # Handle concept relationships
        relationship_dicts = data.pop('concept_relationships', [])
        
        # Create instance
        instance = cls(**data)
        instance.embedding = embedding
        
        # Add concept relationships
        for rel_dict in relationship_dicts:
            instance.concept_relationships.append(ConceptRelationship.from_dict(rel_dict))
        
        return instance
    
    def add_theme(self, theme: str, importance: float = 0.5, confidence: float = 0.8) -> None:
        """Add or update a theme with importance score and confidence"""
        if theme not in self.themes:
            self.themes[theme] = {
                "importance": importance,
                "first_mentioned": self.timestamp,
                "mentions": 1,
                "priority_tier": PriorityTier.get_tier_from_score(importance)
            }
            self.theme_confidences[theme] = confidence
        else:
            self.themes[theme]["mentions"] += 1
            # Increase importance with repeated mentions
            new_importance = min(1.0, self.themes[theme]["importance"] + 0.1)
            self.themes[theme]["importance"] = new_importance
            self.themes[theme]["priority_tier"] = PriorityTier.get_tier_from_score(new_importance)
            # Update confidence with weighted average
            self.theme_confidences[theme] = (self.theme_confidences[theme] * 0.7 + confidence * 0.3)
    
    def add_concept(
        self, 
        concept: str, 
        definition: str, 
        importance: float = 0.5,
        confidence: float = 0.8,
        priority_tier: Optional[str] = None
    ) -> None:
        """Add or update a concept with definition, importance, and confidence"""
        if priority_tier is None:
            priority_tier = PriorityTier.get_tier_from_score(importance)
            
        if concept not in self.concepts:
            self.concepts[concept] = {
                "importance": importance,
                "definition": definition,
                "first_mentioned": self.timestamp,
                "evolution": [],
                "mentions": 1,
                "priority_tier": priority_tier
            }
            self.concept_confidences[concept] = confidence
        else:
            prev_def = self.concepts[concept]["definition"]
            if prev_def != definition:
                # Track concept evolution with confidence information
                self.concepts[concept]["evolution"].append({
                    "timestamp": self.timestamp,
                    "previous_definition": prev_def,
                    "new_definition": definition,
                    "confidence": confidence,
                    "evolution_type": self._determine_evolution_type(prev_def, definition)
                })
                self.concepts[concept]["definition"] = definition
            
            self.concepts[concept]["mentions"] += 1
            # Increase importance with repeated mentions
            new_importance = min(1.0, self.concepts[concept]["importance"] + 0.1)
            self.concepts[concept]["importance"] = new_importance
            
            # Update tier if it changed
            current_tier = self.concepts[concept].get("priority_tier", PriorityTier.ACTIVE)
            if priority_tier != current_tier and PriorityTier.get_importance_score(priority_tier) > PriorityTier.get_importance_score(current_tier):
                # Only upgrade tiers, never downgrade
                self.concepts[concept]["priority_tier"] = priority_tier
            
            # Update confidence with weighted average (more weight to previous)
            self.concept_confidences[concept] = (self.concept_confidences[concept] * 0.7 + confidence * 0.3)
    
    def _determine_evolution_type(self, prev_def: str, new_def: str) -> str:
        """Determine the type of concept evolution"""
        if len(new_def) > len(prev_def) * 1.5:
            return "expansion"
        elif len(new_def) < len(prev_def) * 0.7:
            return "reduction"
        else:
            return "refinement"
    
    def add_relationship(
        self, 
        source_concept: str, 
        target_concept: str, 
        relationship_type: str,
        strength: float = 0.7,
        bidirectional: bool = False,
        description: str = "",
        confidence: float = 0.8
    ) -> None:
        """Add a relationship between two concepts"""
        relationship = ConceptRelationship(
            source_concept=source_concept,
            target_concept=target_concept,
            relationship_type=relationship_type,
            strength=strength,
            bidirectional=bidirectional,
            description=description,
            first_mentioned=self.timestamp,
            entry_id=self.id,
            confidence=confidence
        )
        
        # Check if this relationship already exists (handles bidirectional matching)
        if relationship not in self.concept_relationships:
            self.concept_relationships.append(relationship)
    
    def add_child(self, child_id: str) -> None:
        """Add a child entry ID to establish conversation flow"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def mark_validated(self, validation_notes: str = None) -> None:
        """Mark this entry as validated for logical consistency"""
        self.validated = True
        self.validation_notes = validation_notes
    
    def get_low_confidence_concepts(self, threshold: float = 0.7) -> List[str]:
        """Get concepts with confidence below threshold"""
        return [concept for concept, confidence in self.concept_confidences.items() 
                if confidence < threshold]
    
    def get_concepts_by_tier(self, tier: str) -> List[str]:
        """Get concepts of a specific priority tier"""
        return [concept for concept, metadata in self.concepts.items() 
                if metadata.get("priority_tier", PriorityTier.ACTIVE) == tier]
    
    def get_core_concepts(self) -> List[str]:
        """Get all core concepts"""
        return self.get_concepts_by_tier(PriorityTier.CORE)

# ============= Concept Knowledge Graph Manager =============

class ConceptGraphManager:
    """
    Manages the concept relationship graph across conversation entries
    
    This component:
    1. Tracks all concept relationships in a unified graph
    2. Enables querying for related concepts
    3. Calculates concept centrality and importance
    4. Detects emerging concept clusters
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.relationships: Dict[Tuple[str, str, str], ConceptRelationship] = {}
        self.concept_to_relationships: Dict[str, List[ConceptRelationship]] = defaultdict(list)
        
        # Graph analytics data
        self.concept_centrality: Dict[str, float] = {}  # Concept -> centrality score
        self.concept_clusters: List[Set[str]] = []  # Sets of related concepts
        
        logger.info("ConceptGraphManager initialized")
    
    def add_relationship(self, relationship: ConceptRelationship) -> None:
        """Add a relationship to the graph"""
        key = (relationship.source_concept, relationship.target_concept, relationship.relationship_type)
        
        if key in self.relationships:
            # Update existing relationship
            existing = self.relationships[key]
            
            # Update with higher confidence/strength if applicable
            if relationship.confidence > existing.confidence:
                existing.confidence = relationship.confidence
                
            if relationship.strength > existing.strength:
                existing.strength = relationship.strength
                
            # Update description if provided
            if relationship.description and not existing.description:
                existing.description = relationship.description
        else:
            # Add new relationship
            self.relationships[key] = relationship
            
            # Update concept_to_relationships index
            self.concept_to_relationships[relationship.source_concept].append(relationship)
            if relationship.bidirectional:
                self.concept_to_relationships[relationship.target_concept].append(relationship)
            else:
                # Even for directed relationships, we want to track that the target is involved
                target_rel = ConceptRelationship(
                    source_concept=relationship.target_concept,
                    target_concept=relationship.source_concept,
                    relationship_type=f"inverse_{relationship.relationship_type}",
                    strength=relationship.strength * 0.8,  # Slightly lower strength
                    bidirectional=False,
                    description=f"Inverse of: {relationship.description}" if relationship.description else "",
                    first_mentioned=relationship.first_mentioned,
                    entry_id=relationship.entry_id,
                    confidence=relationship.confidence * 0.9  # Slightly lower confidence
                )
                self.concept_to_relationships[relationship.target_concept].append(target_rel)
        
        # Refresh graph analytics
        self._update_concept_centrality()
    
    def add_relationships_from_entry(self, entry: ConversationEntry) -> None:
        """Add all relationships from an entry"""
        for relationship in entry.concept_relationships:
            self.add_relationship(relationship)
    
    def get_related_concepts(self, concept: str, min_strength: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Get concepts related to the given concept
        
        Returns:
            List of (related_concept, relationship_type, strength) tuples
        """
        related = []
        
        for rel in self.concept_to_relationships[concept]:
            # Get the related concept (the one that's not the input concept)
            related_concept = rel.target_concept if rel.source_concept == concept else rel.source_concept
            
            # Only include relationships above the minimum strength
            if rel.strength >= min_strength:
                related.append((related_concept, rel.relationship_type, rel.strength))
        
        # Sort by descending strength
        return sorted(related, key=lambda x: x[2], reverse=True)
    
    def get_concept_importance(self, concept: str) -> float:
        """
        Calculate concept importance based on centrality and relationship strength
        
        Returns:
            Importance score (0.0-1.0)
        """
        # Start with centrality (or 0.0 if not calculated)
        importance = self.concept_centrality.get(concept, 0.0)
        
        # Add weighted component based on relationship strength
        total_strength = sum(rel.strength for rel in self.concept_to_relationships[concept])
        num_relationships = len(self.concept_to_relationships[concept])
        
        if num_relationships > 0:
            avg_strength = total_strength / num_relationships
            # Weight centrality and average strength
            importance = 0.7 * importance + 0.3 * avg_strength
        
        return min(1.0, importance)
    
    def extract_relationships(self, entry: ConversationEntry) -> None:
        """
        Extract concept relationships from an entry using LLM
        """
        if not self.llm:
            logger.warning("No LLM provided, skipping relationship extraction")
            return
            
        concepts = list(entry.concepts.keys())
        if len(concepts) < 2:
            return
            
        # Create concept pairs for analysis (limit to avoid too many API calls)
        import itertools
        concept_pairs = list(itertools.combinations(concepts[:7], 2))  # Limit to 7 concepts to avoid too many combinations
        
        for source, target in concept_pairs:
            source_def = entry.concepts[source].get("definition", "")
            target_def = entry.concepts[target].get("definition", "")
            
            # Skip if either definition is missing
            if not source_def or not target_def:
                continue
                
            # Use LLM to analyze relationship
            prompt = f"""
            Analyze the relationship between these two concepts from a conversation:
            
            Concept 1: {source}
            Definition: {source_def}
            
            Concept 2: {target}
            Definition: {target_def}
            
            Does a meaningful relationship exist between these concepts? If yes, specify:
            1. Relationship type (one of: depends_on, refines, conflicts_with, is_similar_to, contains, implements, contradicts)
            2. Is this relationship bidirectional (true/false)
            3. Relationship strength (0.0-1.0)
            4. Brief description of the relationship (up to 10 words)
            5. Confidence (0.0-1.0) that this relationship truly exists
            
            Return your analysis as a JSON object with fields: relationship_exists (boolean), relationship_type, bidirectional, strength, description, confidence
            If no relationship exists, just return: {{"relationship_exists": false}}
            """
            
            try:
                response = self.llm.predict(prompt)
                
                # Parse JSON response
                import re, json
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    relationship_data = json.loads(json_match.group(0))
                    
                    if relationship_data.get("relationship_exists", False):
                        # Add relationship to entry
                        entry.add_relationship(
                            source_concept=source,
                            target_concept=target,
                            relationship_type=relationship_data.get("relationship_type", "related_to"),
                            strength=relationship_data.get("strength", 0.5),
                            bidirectional=relationship_data.get("bidirectional", False),
                            description=relationship_data.get("description", ""),
                            confidence=relationship_data.get("confidence", 0.7)
                        )
            except Exception as e:
                logger.warning(f"Failed to extract relationship: {e}")
    
    def _update_concept_centrality(self) -> None:
        """
        Update concept centrality by calculating a simplified version of PageRank
        """
        # Only recalculate if we have a significant number of relationships
        if len(self.relationships) < 5:
            # Simple centrality based on relationship count
            for concept in self.concept_to_relationships:
                self.concept_centrality[concept] = min(1.0, len(self.concept_to_relationships[concept]) / 5.0)
            return
            
        # Simplified PageRank calculation
        # Initialize with equal centrality
        all_concepts = set()
        for rel in self.relationships.values():
            all_concepts.add(rel.source_concept)
            all_concepts.add(rel.target_concept)
        
        centrality = {concept: 1.0 / len(all_concepts) for concept in all_concepts}
        damping = 0.85
        iterations = 5
        
        # Iterative update
        for _ in range(iterations):
            new_centrality = {concept: (1.0 - damping) / len(all_concepts) for concept in all_concepts}
            
            for concept in all_concepts:
                # Find all concepts that link to this one
                incoming = []
                for rel_key, rel in self.relationships.items():
                    if rel.target_concept == concept:
                        incoming.append((rel.source_concept, rel.strength))
                    elif rel.bidirectional and rel.source_concept == concept:
                        incoming.append((rel.target_concept, rel.strength))
                
                # Update centrality
                for source, strength in incoming:
                    outgoing_count = len(self.concept_to_relationships[source])
                    if outgoing_count > 0:
                        new_centrality[concept] += damping * centrality[source] * strength / outgoing_count
            
            # Normalize
            sum_centrality = sum(new_centrality.values())
            if sum_centrality > 0:
                for concept in new_centrality:
                    new_centrality[concept] /= sum_centrality
            
            centrality = new_centrality
        
        self.concept_centrality = centrality
        
        # Identify concept clusters
        self._identify_concept_clusters()
    
    def _identify_concept_clusters(self) -> None:
        """
        Identify clusters of related concepts using a simple clustering algorithm
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        for rel_key, rel in self.relationships.items():
            if rel.strength >= 0.6:  # Only use strong relationships
                adjacency[rel.source_concept].add(rel.target_concept)
                if rel.bidirectional:
                    adjacency[rel.target_concept].add(rel.source_concept)
        
        # Run simplified connected components algorithm
        visited = set()
        clusters = []
        
        for concept in adjacency:
            if concept in visited:
                continue
                
            # Start a new cluster with BFS
            cluster = set()
            queue = [concept]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                cluster.add(current)
                
                # Add neighbors
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(cluster) > 1:  # Only add non-trivial clusters
                clusters.append(cluster)
        
        self.concept_clusters = clusters
    
    def visualize_graph(self) -> str:
        """
        Generate a Mermaid graph visualization of the concept network
        
        Returns:
            Mermaid graph definition
        """
        if not self.relationships:
            return "graph TD\n    A[No relationships yet]"
            
        mermaid = ["graph TD"]
        
        # Add nodes
        all_concepts = set()
        for rel in self.relationships.values():
            all_concepts.add(rel.source_concept)
            all_concepts.add(rel.target_concept)
        
        for concept in all_concepts:
            # Create node ID (replace spaces and special chars)
            node_id = concept.replace(" ", "_").replace("-", "_")
            importance = self.get_concept_importance(concept)
            
            # Style based on importance
            if importance >= 0.7:
                mermaid.append(f'    {node_id}["{concept}"] ::highImportance')
            elif importance >= 0.4:
                mermaid.append(f'    {node_id}["{concept}"] ::mediumImportance')
            else:
                mermaid.append(f'    {node_id}["{concept}"]')
        
        # Add relationships
        for rel in self.relationships.values():
            source_id = rel.source_concept.replace(" ", "_").replace("-", "_")
            target_id = rel.target_concept.replace(" ", "_").replace("-", "_")
            
            # Style based on relationship type and strength
            edge_style = "-->"
            if rel.relationship_type == "conflicts_with" or rel.relationship_type == "contradicts":
                edge_style = "-.->"
            elif rel.relationship_type == "is_similar_to":
                edge_style = "---"
            elif rel.bidirectional:
                edge_style = "<-->"
                
            # Add strength to line width
            if rel.strength >= 0.7:
                edge_style = edge_style.replace("-", "=")
                
            # Add the relationship
            label = f"|{rel.relationship_type}|" if rel.relationship_type else ""
            mermaid.append(f"    {source_id} {edge_style} {target_id} {label}")
        
        # Add styling
        mermaid.append("    classDef highImportance fill:#f96,stroke:#333,stroke-width:2px")
        mermaid.append("    classDef mediumImportance fill:#fc9,stroke:#333")
        
        return "\n".join(mermaid)

# ============= Adaptive Context Prioritizer =============

class AdaptiveContextPrioritizer:
    """
    Dynamically adjusts concept and theme priorities based on usage patterns
    
    This component:
    1. Tracks "temperature" of concepts based on recency
    2. Adjusts importance based on query relevance
    3. Allows dynamic priority tier adjustments
    """
    def __init__(self, context_tracker: 'SequentialContextTracker'):
        self.context_tracker = context_tracker
        
        # Temperature tracking (recency of usage)
        self.concept_temperature = {}  # concept -> temperature
        self.theme_temperature = {}  # theme -> temperature
        
        # Decay settings
        self.decay_factor = 0.9  # Temperature decay with each turn
        self.temperature_boost = 1.0  # Boost when mentioned
        
        logger.info("AdaptiveContextPrioritizer initialized")
    
    def update_temperatures(self, current_entry: ConversationEntry):
        """Update concept temperatures based on current conversation entry"""
        # Decay all existing temperatures
        for concept in self.concept_temperature:
            self.concept_temperature[concept] *= self.decay_factor
            
        for theme in self.theme_temperature:
            self.theme_temperature[theme] *= self.decay_factor
            
        # Boost temperature for concepts in current entry
        for concept in current_entry.concepts:
            self.concept_temperature[concept] = self.temperature_boost
            
        # Boost temperature for themes in current entry
        for theme in current_entry.themes:
            self.theme_temperature[theme] = self.temperature_boost
    
    def get_prioritized_concepts(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get concepts prioritized by importance and temperature"""
        # Get high importance concepts
        important_concepts = self.context_tracker.get_high_importance_concepts(threshold=threshold)
        
        # Apply temperature boosting
        for concept in important_concepts:
            concept_name = concept["concept"]
            temp = self.concept_temperature.get(concept_name, 0.0)
            
            # Blend importance with temperature (adjustable formula)
            blended_score = 0.7 * concept["importance"] + 0.3 * temp
            concept["blended_score"] = blended_score
            
            # Update priority tier if needed
            original_tier = concept.get("priority_tier", PriorityTier.ACTIVE)
            if temp > 0.7 and original_tier == PriorityTier.BACKGROUND:
                # Temporarily elevate background concepts that are currently hot
                concept["priority_tier"] = PriorityTier.ACTIVE
                concept["tier_boosted"] = True
            
        # Sort by blended score
        return sorted(important_concepts, key=lambda c: c["blended_score"], reverse=True)
    
    def adjust_concept_priority(self, concept: str, new_tier: str) -> bool:
        """
        Manually adjust priority tier for a concept
        
        Returns:
            True if successful, False if concept not found
        """
        # Check if concept exists in registry
        if concept in self.context_tracker.concept_registry:
            # Update priority tier
            self.context_tracker.concept_registry[concept]["priority_tier"] = new_tier
            
            # Update importance to match tier
            tier_importance = PriorityTier.get_importance_score(new_tier)
            self.context_tracker.concept_registry[concept]["importance"] = max(
                self.context_tracker.concept_registry[concept]["importance"],
                tier_importance
            )
            
            return True
        
        return False

# ============= Conceptual Diff Algorithm =============

class ConceptualDiffAlgorithm:
    """
    Analyzes how concepts evolve and determines appropriate merging strategy
    
    This algorithm:
    1. Compares new definitions to existing ones
    2. Classifies changes as expansion, correction, or reinforcement
    3. Recommends proper merging strategy
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        logger.info("ConceptualDiffAlgorithm initialized")
    
    def compare_concept_definitions(
        self, 
        concept: str, 
        existing_definition: str, 
        new_definition: str
    ) -> Dict[str, Any]:
        """
        Compare existing and new concept definitions to determine evolution type
        
        Returns:
            Dictionary with evolution analysis
        """
        # Simple heuristic-based comparison if no LLM available
        if not self.llm:
            return self._heuristic_comparison(existing_definition, new_definition)
        
        # Use LLM for more sophisticated comparison
        prompt = f"""
        Analyze how this concept has evolved by comparing the existing definition with a new one:
        
        CONCEPT: {concept}
        
        EXISTING DEFINITION:
        {existing_definition}
        
        NEW DEFINITION:
        {new_definition}
        
        Determine the type of evolution:
        1. "expansion" - New definition adds depth or breadth to existing definition
        2. "correction" - New definition corrects or contradicts existing definition
        3. "refinement" - New definition clarifies or slightly modifies existing definition
        4. "reinforcement" - New definition essentially restates existing definition
        
        Also determine:
        - Confidence score (0.0-1.0) in your classification
        - Key additions in the new definition
        - Key removals from the existing definition
        - Which definition is more precise
        - Integration strategy (merge, replace, or keep separate)
        
        Return your analysis as a JSON object.
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Parse JSON response
            import re, json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                return analysis
        except Exception as e:
            logger.warning(f"Failed to analyze concept evolution using LLM: {e}")
            
        # Fallback to heuristic comparison
        return self._heuristic_comparison(existing_definition, new_definition)
    
    def _heuristic_comparison(self, existing_definition: str, new_definition: str) -> Dict[str, Any]:
        """
        Simple heuristic-based comparison as fallback
        """
        # Simple analysis based on length comparison
        existing_words = set(existing_definition.lower().split())
        new_words = set(new_definition.lower().split())
        
        # Calculate overlapping words
        common_words = existing_words.intersection(new_words)
        only_in_existing = existing_words - new_words
        only_in_new = new_words - existing_words
        
        # Calculate similarity ratio
        if len(existing_words) == 0 and len(new_words) == 0:
            similarity = 1.0
        elif len(existing_words) == 0:
            similarity = 0.0
        else:
            similarity = len(common_words) / max(len(existing_words), len(new_words))
        
        # Determine evolution type
        evolution_type = "reinforcement"  # Default
        
        if similarity < 0.3:
            evolution_type = "correction"
        elif similarity < 0.7:
            if len(new_words) > len(existing_words) * 1.3:
                evolution_type = "expansion"
            else:
                evolution_type = "refinement"
        
        # Determine integration strategy
        if evolution_type == "correction":
            strategy = "replace"
        elif evolution_type == "expansion":
            strategy = "merge"
        elif evolution_type == "refinement":
            strategy = "merge"
        else:  # reinforcement
            strategy = "keep"
        
        return {
            "evolution_type": evolution_type,
            "confidence": 0.7,
            "additions": list(only_in_new)[:5],  # Limit to 5 words
            "removals": list(only_in_existing)[:5],  # Limit to 5 words
            "more_precise": "new" if len(new_definition) > len(existing_definition) * 1.2 else "existing",
            "integration_strategy": strategy,
            "similarity": similarity
        }
    
    def merge_definitions(
        self, 
        concept: str, 
        existing_definition: str, 
        new_definition: str,
        evolution_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Merge existing and new concept definitions based on evolution analysis
        
        Returns:
            Merged definition
        """
        # Get evolution analysis if not provided
        if not evolution_analysis:
            evolution_analysis = self.compare_concept_definitions(
                concept, existing_definition, new_definition
            )
        
        integration_strategy = evolution_analysis.get("integration_strategy", "merge")
        
        # Apply integration strategy
        if integration_strategy == "replace":
            return new_definition
        elif integration_strategy == "keep":
            return existing_definition
        elif integration_strategy == "merge":
            # Attempt to merge using LLM if available
            if self.llm:
                merge_prompt = f"""
                Merge these two definitions of the same concept into a unified definition:
                
                CONCEPT: {concept}
                
                EXISTING DEFINITION:
                {existing_definition}
                
                NEW DEFINITION:
                {new_definition}
                
                EVOLUTION TYPE: {evolution_analysis.get('evolution_type', 'unknown')}
                
                Create a single coherent definition that preserves:
                1. All key information from both definitions
                2. Resolves any contradictions (favoring the newer definition)
                3. Maintains clear, concise language
                
                Return only the merged definition, nothing else.
                """
                
                try:
                    merged = self.llm.predict(merge_prompt).strip()
                    return merged
                except Exception as e:
                    logger.warning(f"Failed to merge definitions using LLM: {e}")
            
            # Fallback simple merging
            if len(existing_definition) > len(new_definition):
                return f"{existing_definition} {new_definition}"
            else:
                return f"{new_definition} Additionally: {existing_definition}"
        
        # Default fallback
        return new_definition

# ============= Sequential Context Tracker =============

class SequentialContextTracker:
    """
    Tracks conversation in logical sequence with hierarchical importance
    to prevent context loss and ensure logical flow
    """
    def __init__(
        self, 
        pg_connection_string: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm: Optional[Any] = None
    ):
        self.pg_conn_string = pg_connection_string
        self.llm = llm
        
        # Setup embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Setup vector store
        self.vector_store = PGVector(
            connection_string=pg_connection_string,
            embedding_function=self.embeddings,
            collection_name="context_vectors"
        )
        
        # Conversation graph for tracking logical flow
        self.conversation_graph: Dict[str, List[str]] = defaultdict(list)  # parent_id -> [child_ids]
        self.entry_cache: Dict[str, ConversationEntry] = {}  # id -> entry
        
        # Theme and concept registries for tracking importance across conversation
        self.theme_registry: Dict[str, Dict[str, Any]] = {}  # theme -> metadata
        self.concept_registry: Dict[str, Dict[str, Any]] = {}  # concept -> metadata
        
        # Root node for conversation tree
        self.root_node_id: Optional[str] = None
        
        # Initialize enhanced components
        self.concept_graph = ConceptGraphManager(llm)
        self.context_prioritizer = AdaptiveContextPrioritizer(self)
        self.concept_diff = ConceptualDiffAlgorithm(llm)
        
        logger.info("SequentialContextTracker initialized with enhanced components")
    
    def add_entry(self, entry: ConversationEntry, parent_id: Optional[str] = None) -> str:
        """
        Add a conversation entry with logical parent-child relationship
        Returns entry ID
        """
        # Set parent ID
        if parent_id:
            entry.parent_id = parent_id
            
            # Update parent's child list
            parent_entry = self.get_entry(parent_id)
            if parent_entry:
                parent_entry.add_child(entry.id)
                self._update_entry(parent_entry)
                
            # Update conversation graph
            self.conversation_graph[parent_id].append(entry.id)
        elif not self.root_node_id:
            # This is the first entry in the conversation
            self.root_node_id = entry.id
        
        # Generate embedding if needed
        if entry.embedding is None and (entry.user_input or entry.system_response):
            combined_text = f"{entry.user_input} {entry.system_response}"
            entry.embedding = self.embeddings.embed_query(combined_text)
        
        # Extract concept relationships if LLM is available
        if self.llm and entry.concepts and len(entry.concepts) > 1:
            self.concept_graph.extract_relationships(entry)
        
        # Add relationships from this entry to the graph
        self.concept_graph.add_relationships_from_entry(entry)
            
        # Update theme and concept registries
        self._update_registries(entry)
        
        # Update concept temperatures
        self.context_prioritizer.update_temperatures(entry)
        
        # Store entry
        self._store_entry(entry)
        
        # Cache entry
        self.entry_cache[entry.id] = entry
        
        return entry.id
    
    def _update_registries(self, entry: ConversationEntry) -> None:
        """Update theme and concept registries with entry data"""
        # Update theme registry
        for theme, metadata in entry.themes.items():
            if theme not in self.theme_registry:
                self.theme_registry[theme] = {
                    "importance": metadata["importance"],
                    "first_mentioned": metadata["first_mentioned"],
                    "mentions": metadata["mentions"],
                    "entries": [entry.id],
                    "priority_tier": metadata.get("priority_tier", PriorityTier.ACTIVE)
                }
            else:
                self.theme_registry[theme]["mentions"] += metadata["mentions"]
                self.theme_registry[theme]["importance"] = max(
                    self.theme_registry[theme]["importance"],
                    metadata["importance"]
                )
                
                # Update tier if higher priority
                current_tier = self.theme_registry[theme].get("priority_tier", PriorityTier.ACTIVE)
                new_tier = metadata.get("priority_tier", PriorityTier.ACTIVE)
                
                if PriorityTier.get_importance_score(new_tier) > PriorityTier.get_importance_score(current_tier):
                    self.theme_registry[theme]["priority_tier"] = new_tier
                
                if entry.id not in self.theme_registry[theme]["entries"]:
                    self.theme_registry[theme]["entries"].append(entry.id)
        
        # Update concept registry with intelligent merging
        for concept, metadata in entry.concepts.items():
            if concept not in self.concept_registry:
                # New concept
                self.concept_registry[concept] = {
                    "importance": metadata["importance"],
                    "definition": metadata["definition"],
                    "first_mentioned": metadata["first_mentioned"],
                    "evolution": metadata.get("evolution", []).copy(),
                    "mentions": metadata["mentions"],
                    "entries": [entry.id],
                    "priority_tier": metadata.get("priority_tier", PriorityTier.ACTIVE)
                }
            else:
                # Update existing concept
                self.concept_registry[concept]["mentions"] += metadata["mentions"]
                
                # Update importance (use higher value)
                self.concept_registry[concept]["importance"] = max(
                    self.concept_registry[concept]["importance"],
                    metadata["importance"]
                )
                
                # Update tier if higher priority
                current_tier = self.concept_registry[concept].get("priority_tier", PriorityTier.ACTIVE)
                new_tier = metadata.get("priority_tier", PriorityTier.ACTIVE)
                
                if PriorityTier.get_importance_score(new_tier) > PriorityTier.get_importance_score(current_tier):
                    self.concept_registry[concept]["priority_tier"] = new_tier
                
                # Apply conceptual diff algorithm to handle definition evolution
                current_def = self.concept_registry[concept]["definition"]
                new_def = metadata.get("definition", "")
                
                if new_def and new_def != current_def:
                    # Analyze how concept has evolved
                    evolution_analysis = self.concept_diff.compare_concept_definitions(
                        concept, current_def, new_def
                    )
                    
                    # Record evolution
                    evolution_record = {
                        "timestamp": entry.timestamp,
                        "previous_definition": current_def,
                        "new_definition": new_def,
                        "entry_id": entry.id,
                        "evolution_type": evolution_analysis.get("evolution_type", "unknown"),
                        "confidence": evolution_analysis.get("confidence", 0.7)
                    }
                    self.concept_registry[concept]["evolution"].append(evolution_record)
                    
                    # Apply merging strategy
                    merged_definition = self.concept_diff.merge_definitions(
                        concept, current_def, new_def, evolution_analysis
                    )
                    
                    # Update definition
                    self.concept_registry[concept]["definition"] = merged_definition
                    self.concept_registry[concept]["last_updated"] = entry.timestamp
                
                # Add entry to tracking
                if entry.id not in self.concept_registry[concept]["entries"]:
                    self.concept_registry[concept]["entries"].append(entry.id)
    
    def _store_entry(self, entry: ConversationEntry) -> None:
        """Store entry in vector store and update cache"""
        # Store vector in database
        if entry.embedding is not None:
            self.vector_store.add_texts(
                texts=[f"{entry.user_input} {entry.system_response}"],
                metadatas=[{"id": entry.id}],
                embeddings=[entry.embedding]
            )
        
        # Store full entry in cache
        self.entry_cache[entry.id] = entry
    
    def _update_entry(self, entry: ConversationEntry) -> None:
        """Update an existing entry"""
        # Update cache
        self.entry_cache[entry.id] = entry
        
        # Update vector store (would require delete and re-add in actual implementation)
        if entry.embedding is not None:
            # This is simplified - in practice you'd need to delete and re-add
            self.vector_store.add_texts(
                texts=[f"{entry.user_input} {entry.system_response}"],
                metadatas=[{"id": entry.id}],
                embeddings=[entry.embedding]
            )
    
    def get_entry(self, entry_id: str) -> Optional[ConversationEntry]:
        """Retrieve an entry by ID"""
        # Check cache first
        if entry_id in self.entry_cache:
            return self.entry_cache[entry_id]
        
        # Otherwise would fetch from database
        # (Simplified implementation)
        return None
    
    def get_conversation_path(self, entry_id: str) -> List[ConversationEntry]:
        """
        Get the logical path from root to the specified entry
        to maintain contextual continuity
        """
        path = []
        current_id = entry_id
        
        # Traverse upward to root
        while current_id:
            entry = self.get_entry(current_id)
            if not entry:
                break
                
            path.append(entry)
            current_id = entry.parent_id
        
        # Reverse to get root-to-leaf order
        return list(reversed(path))
    
    def find_sequential_entries(
        self, 
        query: str, 
        top_k: int = 5,
        include_sequential_context: bool = True,
        prioritize_core_concepts: bool = True
    ) -> List[ConversationEntry]:
        """
        Find entries most relevant to query while maintaining logical sequence
        
        Args:
            query: Text to search for
            top_k: Number of direct matches to return
            include_sequential_context: Whether to include parent/child context
            prioritize_core_concepts: Whether to ensure entries with core concepts are included
            
        Returns:
            List of entries in logical sequence
        """
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)
        
        # Get vector matches
        results = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k
        )
        
        # Get entry IDs
        entry_ids = [result.metadata.get("id") for result in results if result.metadata.get("id")]
        
        # Get entries
        entries = [self.get_entry(entry_id) for entry_id in entry_ids if entry_id]
        entries = [entry for entry in entries if entry]  # Filter out None
        
        if not include_sequential_context or not entries:
            # Sort by timestamp and return
            return sorted(entries, key=lambda e: e.timestamp)
        
        # Include sequential context (parents and children)
        context_entries = set()
        
        for entry in entries:
            # Add parents (ancestry)
            path = self.get_conversation_path(entry.id)
            context_entries.update(path)
            
            # Add direct children
            for child_id in entry.child_ids:
                child = self.get_entry(child_id)
                if child:
                    context_entries.add(child)
        
        # If prioritizing core concepts, ensure entries with core concepts are included
        if prioritize_core_concepts:
            # Find all entries with core concepts
            core_entries = []
            for concept, metadata in self.concept_registry.items():
                if metadata.get("priority_tier", PriorityTier.ACTIVE) == PriorityTier.CORE:
                    # Get a recent entry containing this concept
                    for entry_id in reversed(metadata.get("entries", [])):
                        entry = self.get_entry(entry_id)
                        if entry:
                            core_entries.append(entry)
                            break
            
            # Add core entries
            context_entries.update(core_entries)
        
        # Convert to list and sort by timestamp
        context_list = sorted(list(context_entries), key=lambda e: e.timestamp)
        
        return context_list
    
    def get_high_importance_concepts(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get concepts that have high importance across the conversation"""
        important_concepts = []
        
        for concept, metadata in self.concept_registry.items():
            # Adjust importance based on graph centrality if concept is in graph
            graph_importance = self.concept_graph.get_concept_importance(concept)
            adjusted_importance = metadata["importance"]
            
            if graph_importance > 0:
                # Blend registry importance with graph-based importance
                adjusted_importance = 0.7 * metadata["importance"] + 0.3 * graph_importance
            
            # Include if above threshold
            if adjusted_importance >= threshold:
                important_concepts.append({
                    "concept": concept,
                    "importance": adjusted_importance,
                    "definition": metadata["definition"],
                    "mentions": metadata["mentions"],
                    "evolution": metadata["evolution"],
                    "priority_tier": metadata.get("priority_tier", PriorityTier.ACTIVE),
                    "graph_importance": graph_importance
                })
        
        # Sort by importance (descending)
        important_concepts.sort(key=lambda c: c["importance"], reverse=True)
        
        return important_concepts
    
    def get_concept_context(self, concept: str) -> Dict[str, Any]:
        """
        Get comprehensive context for a concept
        
        Returns:
            Dictionary with concept details, evolution, and relationships
        """
        if concept not in self.concept_registry:
            return {
                "found": False,
                "concept": concept
            }
        
        # Get basic concept data
        concept_data = self.concept_registry[concept].copy()
        
        # Add relationship data
        related_concepts = self.concept_graph.get_related_concepts(concept)
        concept_data["related_concepts"] = related_concepts
        
        # Get centrality and importance
        concept_data["graph_importance"] = self.concept_graph.get_concept_importance(concept)
        concept_data["centrality"] = self.concept_graph.concept_centrality.get(concept, 0.0)
        
        # Get temperature (recency)
        concept_data["temperature"] = self.context_prioritizer.concept_temperature.get(concept, 0.0)
        
        # Get sample entry IDs (limit to recent entries)
        if concept_data.get("entries"):
            concept_data["recent_entries"] = concept_data["entries"][-3:]
        
        concept_data["found"] = True
        return concept_data
    
    def get_theme_evolution(self) -> Dict[str, List[Dict[str, Any]]]:
        """Track how themes have evolved throughout the conversation"""
        theme_evolution = {}
        
        for theme, metadata in self.theme_registry.items():
            # Get entries that mention this theme, sorted by timestamp
            entries = [self.get_entry(entry_id) for entry_id in metadata["entries"]]
            entries = [e for e in entries if e]  # Filter out None
            entries.sort(key=lambda e: e.timestamp)
            
            # Track mentions over time
            evolution = []
            for entry in entries:
                if theme in entry.themes:
                    evolution.append({
                        "timestamp": entry.timestamp,
                        "importance": entry.themes[theme]["importance"],
                        "entry_id": entry.id
                    })
            
            theme_evolution[theme] = evolution
        
        return theme_evolution
    
    def create_conversation_summary(self, max_length: int = 500) -> str:
        """
        Create a summary of the entire conversation
        
        Args:
            max_length: Maximum summary length in characters
            
        Returns:
            Text summary of the conversation
        """
        if not self.llm:
            # Simple summary if no LLM available
            num_entries = len(self.entry_cache)
            core_concepts = []
            for concept, metadata in self.concept_registry.items():
                if metadata.get("priority_tier") == PriorityTier.CORE:
                    core_concepts.append(concept)
            
            return f"Conversation with {num_entries} entries. Key concepts: {', '.join(core_concepts[:5])}"
        
        # Get all entries in chronological order
        all_entries = sorted(self.entry_cache.values(), key=lambda e: e.timestamp)
        
        # Select important entries (first, last, and core concept entries)
        important_entries = []
        
        # Always include first and last entries
        if all_entries:
            important_entries.append(all_entries[0])
            if len(all_entries) > 1:
                important_entries.append(all_entries[-1])
        
        # Include entries with core concepts (limit to reasonable number)
        core_entries = []
        for concept, metadata in self.concept_registry.items():
            if metadata.get("priority_tier") == PriorityTier.CORE:
                for entry_id in metadata.get("entries", [])[-1:]:  # Just the most recent mention
                    entry = self.get_entry(entry_id)
                    if entry and entry not in important_entries and entry not in core_entries:
                        core_entries.append(entry)
                        
                        # Limit to reasonable number
                        if len(core_entries) >= 3:
                            break
        
        important_entries.extend(core_entries)
        
        # Format entries for summary
        formatted_entries = []
        for i, entry in enumerate(sorted(important_entries, key=lambda e: e.timestamp)):
            formatted_entries.append(f"Entry {i+1}:")
            formatted_entries.append(f"User: {entry.user_input[:100]}...")
            formatted_entries.append(f"System: {entry.system_response[:100]}...")
            formatted_entries.append("")
        
        # Get core concepts
        core_concepts = []
        for concept, metadata in self.concept_registry.items():
            if metadata.get("priority_tier") == PriorityTier.CORE:
                core_concepts.append({
                    "concept": concept,
                    "definition": metadata.get("definition", "")[:100]
                })
        
        # Format core concepts
        formatted_concepts = []
        for concept in core_concepts[:5]:  # Limit to 5 core concepts
            formatted_concepts.append(f"- {concept['concept']}: {concept['definition']}")
        
        # Create summary prompt
        prompt = f"""
        Create a concise summary of this conversation based on key entries and core concepts.
        
        KEY CONVERSATION ENTRIES:
        {chr(10).join(formatted_entries)}
        
        CORE CONCEPTS:
        {chr(10).join(formatted_concepts)}
        
        Please provide a summary that:
        1. Captures the main topics and progression of the conversation
        2. Highlights the core concepts and their relationships
        3. Is approximately {max_length} characters in length
        
        Summary:
        """
        
        # Generate summary
        try:
            summary = self.llm.predict(prompt).strip()
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate conversation summary: {e}")
            return f"Conversation with {len(all_entries)} entries about {', '.join([c['concept'] for c in core_concepts[:3]])}"
    
    def validate_logical_flow(
        self, 
        entry: ConversationEntry, 
        previous_entries: List[ConversationEntry]
    ) -> Tuple[bool, str]:
        """
        Validate that an entry logically follows from previous entries
        
        Returns:
            (valid, reason) tuple
        """
        if not previous_entries:
            # First entry is always valid
            return True, "Initial entry"
        
        # Check for concept consistency with priority tiers
        missing_concepts = {
            PriorityTier.CORE: [],     # Core concepts must never be dropped
            PriorityTier.ACTIVE: [],   # Active concepts should be maintained
            PriorityTier.SUPPORTING: [] # Supporting concepts only when directly relevant
        }
        
        # Get high importance concepts from previous entries
        prev_concepts = {}
        for prev_entry in previous_entries:
            for concept, metadata in prev_entry.concepts.items():
                if concept not in prev_concepts:
                    prev_concepts[concept] = metadata
                else:
                    # Use most recent definition
                    prev_concepts[concept] = metadata
        
        # Current concepts
        current_concepts = set(entry.concepts.keys())
        
        # Check core concepts (must never be dropped)
        for concept, metadata in prev_concepts.items():
            tier = metadata.get("priority_tier", PriorityTier.ACTIVE)
            
            if tier == PriorityTier.CORE and concept not in current_concepts:
                missing_concepts[PriorityTier.CORE].append(concept)
            elif tier == PriorityTier.ACTIVE and concept not in current_concepts:
                missing_concepts[PriorityTier.ACTIVE].append(concept)
        
        # Generate validation result
        if missing_concepts[PriorityTier.CORE]:
            return False, f"Missing core concepts: {', '.join(missing_concepts[PriorityTier.CORE])}"
        elif len(missing_concepts[PriorityTier.ACTIVE]) > 2:
            return False, f"Too many active concepts dropped: {', '.join(missing_concepts[PriorityTier.ACTIVE][:3])}"
            
        return True, "Logically consistent"

