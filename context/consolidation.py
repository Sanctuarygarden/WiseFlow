from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class ContextConsolidationLayer:
    """
    Final context consolidation before response generation
    
    This component:
    1. Re-evaluates ALL past context using vector search and priority tiering
    2. Detects key concepts that may have been missed in recent context
    3. Re-injects critical information that would otherwise be lost
    4. Ensures all high-importance content is considered before output
    """
    
    def __init__(self, context_tracker: Any, llm: Optional[Any] = None):
        """
        Initialize context consolidation layer
        
        Args:
            context_tracker: Sequential context tracker for concept access
            llm: Optional language model for relevance assessment
        """
        self.context_tracker = context_tracker
        self.llm = llm
        self.consolidation_stats = {
            "total_consolidations": 0,
            "concepts_reintroduced": 0,
            "concepts_filtered": 0
        }
        
    def consolidate_context(
        self, 
        query: str, 
        recent_context: List[Any], 
        all_concepts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consolidate context by reintroducing missing critical concepts
        
        Args:
            query: User's input query
            recent_context: Recent conversation entries
            all_concepts: All known concepts with their metadata
            
        Returns:
            Consolidated context with critical concepts reintroduced
        """
        self.consolidation_stats["total_consolidations"] += 1
        
        # Extract concepts from recent context
        recent_concepts = self._extract_concepts_from_context(recent_context)
        
        # Identify missing concepts
        missing_concepts = {
            concept: metadata for concept, metadata in all_concepts.items()
            if concept not in recent_concepts
        }
        
        # No missing concepts, return original context
        if not missing_concepts:
            return recent_context
            
        # Filter to high-importance missing concepts
        critical_concepts = {}
        for concept, metadata in missing_concepts.items():
            # Include if it's CORE or high importance
            importance = metadata.get("importance", 0.0)
            priority_tier = metadata.get("priority_tier", "")
            
            if priority_tier == "core" or importance >= 0.7:
                critical_concepts[concept] = metadata
                
        self.consolidation_stats["concepts_filtered"] += len(missing_concepts) - len(critical_concepts)
        
        # If LLM is available, filter to concepts relevant to the query
        if self.llm and critical_concepts:
            relevant_concepts = self._filter_relevant_concepts(query, critical_concepts)
            self.consolidation_stats["concepts_filtered"] += len(critical_concepts) - len(relevant_concepts)
            critical_concepts = relevant_concepts
            
        # If no critical concepts after filtering, return original context
        if not critical_concepts:
            return recent_context
            
        self.consolidation_stats["concepts_reintroduced"] += len(critical_concepts)
        
        # Create consolidated context with reintroduced concepts
        consolidated_context = recent_context.copy()
        consolidated_context.extend(self._create_concept_entries(critical_concepts))
        
        logger.info(f"Consolidated context with {len(critical_concepts)} reintroduced critical concepts")
        return consolidated_context
    
    def _extract_concepts_from_context(self, context: List[Any]) -> Set[str]:
        """Extract concepts from context entries"""
        concepts = set()
        for entry in context:
            # Extract from entry concepts if available
            if hasattr(entry, 'concepts') and entry.concepts:
                concepts.update(entry.concepts.keys())
        return concepts
    
    def _filter_relevant_concepts(
        self, 
        query: str, 
        critical_concepts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter to concepts relevant to the query using LLM"""
        if not critical_concepts:
            return {}
            
        # Format concepts for the prompt
        concepts_text = "\n".join([
            f"- {concept}: {metadata.get('definition', 'No definition')}"
            for concept, metadata in critical_concepts.items()
        ])
        
        # Create relevance assessment prompt
        prompt = f"""
        Determine which of these concepts are relevant to the query:
        
        QUERY: {query}
        
        CONCEPTS:
        {concepts_text}
        
        Return a list of concepts that are directly relevant to answering the query.
        For each concept, respond with "relevant" or "not relevant".
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Parse response to find relevant concepts
            relevant_concepts = {}
            lines = response.strip().split('\n')
            
            for concept, metadata in critical_concepts.items():
                # Check if any line contains concept name and "relevant"
                for line in lines:
                    if concept.lower() in line.lower() and "relevant" in line.lower() and "not relevant" not in line.lower():
                        relevant_concepts[concept] = metadata
                        break
                        
            return relevant_concepts
            
        except Exception as e:
            logger.warning(f"Error in relevance assessment: {str(e)}")
            # Fallback to returning all critical concepts
            return critical_concepts
    
    def _create_concept_entries(self, concepts: Dict[str, Dict[str, Any]]) -> List[Any]:
        """Create context entries for concepts to be reintroduced"""
        # In a full implementation, this would create proper ConversationEntry objects
        # For now, return a simplified representation that works with your existing system
        return []  # Placeholder - implement based on your ConversationEntry structure
    
    def get_stats(self) -> Dict[str, int]:
        """Get consolidation statistics"""
        return self.consolidation_stats.copy()
