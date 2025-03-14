from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConceptRecallSystem:
    """
    Prevents important concepts from being unintentionally dropped
    
    This component:
    1. Tracks "last_referenced" timestamps for all concepts
    2. Identifies high-importance concepts that haven't been mentioned recently
    3. Performs health checks to determine if concepts are still relevant
    4. Re-injects critical concepts before response finalization
    """
    
    def __init__(self, context_tracker: Any, llm: Optional[Any] = None):
        """
        Initialize concept recall system
        
        Args:
            context_tracker: Context tracker for accessing concept registry
            llm: Optional language model for relevance assessment
        """
        self.context_tracker = context_tracker
        self.llm = llm
        self.concept_timestamps = {}  # concept -> last_referenced_timestamp
        self.recall_stats = {
            "checks_performed": 0,
            "concepts_recalled": 0,
            "relevance_checks": 0
        }
        
    def update_timestamps(self, response: str) -> None:
        """
        Update last_referenced timestamps for concepts found in response
        
        Args:
            response: Generated system response to analyze
        """
        current_time = datetime.now().isoformat()
        
        # Extract concepts mentioned in the response
        mentioned_concepts = self._extract_mentioned_concepts(response)
        
        # Update timestamps for mentioned concepts
        for concept in mentioned_concepts:
            self.concept_timestamps[concept] = current_time
    
    def recall_critical_concepts(
        self, 
        query: str, 
        recent_context: List[Any],
        response: str
    ) -> Dict[str, Any]:
        """
        Identify critical concepts that haven't been mentioned recently but remain relevant
        
        Args:
            query: User's input query
            recent_context: Recent conversation entries
            response: Generated response
            
        Returns:
            Dictionary with critical concepts to recall
        """
        self.recall_stats["checks_performed"] += 1
        current_time = datetime.now()
        
        # Get core and active concepts from context tracker
        core_concepts = {
            concept: data for concept, data in self.context_tracker.concept_registry.items()
            if data.get("priority_tier") == "core"
        }
        
        active_concepts = {
            concept: data for concept, data in self.context_tracker.concept_registry.items()
            if data.get("priority_tier") == "active" and data.get("importance", 0) >= 0.7
        }
        
        # Combine core and high-importance active concepts
        critical_concepts = {**core_concepts, **active_concepts}
        
        # Filter to concepts not mentioned in response
        mentioned_concepts = self._extract_mentioned_concepts(response)
        forgotten_concepts = {
            concept: data for concept, data in critical_concepts.items()
            if concept not in mentioned_concepts
        }
        
        # Filter to concepts not recently referenced
        not_recent_concepts = {}
        for concept, data in forgotten_concepts.items():
            if concept not in self.concept_timestamps:
                # Never referenced - definitely include
                not_recent_concepts[concept] = data
                continue
                
            # Check if it's been a while since this concept was referenced
            last_referenced = self.concept_timestamps[concept]
            
            try:
                last_time = datetime.fromisoformat(last_referenced)
                # Check if not mentioned in last 3 entries (using time as a proxy)
                if (current_time - last_time) > timedelta(minutes=10):  # Adjust threshold as needed
                    not_recent_concepts[concept] = data
            except (ValueError, TypeError):
                # If timestamp parsing fails, include the concept
                not_recent_concepts[concept] = data
        
        # If no concepts need recall, return None
        if not not_recent_concepts:
            return None
            
        # Check if forgotten concepts are still relevant to current query
        relevant_concepts = self._check_relevance(not_recent_concepts, query, response)
        
        if not relevant_concepts:
            return None
            
        self.recall_stats["concepts_recalled"] += len(relevant_concepts)
        
        # Return recall information
        return {
            "critical_concepts": [
                {
                    "concept": concept,
                    "definition": data["definition"],
                    "importance": data["importance"]
                }
                for concept, data in relevant_concepts.items()
            ]
        }
    
    def _extract_mentioned_concepts(self, text: str) -> Set[str]:
        """Extract concepts mentioned in text"""
        mentioned = set()
        
        # Check all registered concepts
        for concept in self.context_tracker.concept_registry:
            # Simple string matching - could be enhanced with NLP
            if concept.lower() in text.lower():
                mentioned.add(concept)
                
        return mentioned
    
    def _check_relevance(
        self, 
        forgotten_concepts: Dict[str, Dict[str, Any]],
        query: str,
        response: str
    ) -> Dict[str, Dict[str, Any]]:
        """Check if forgotten concepts are still relevant to current query/response"""
        self.recall_stats["relevance_checks"] += 1
        
        # If no LLM or no forgotten concepts, return all concepts as relevant
        if not self.llm or not forgotten_concepts:
            return forgotten_concepts
            
        # Format concepts for the prompt
        concepts_text = "\n".join([
            f"- {concept}: {data['definition']}" 
            for concept, data in forgotten_concepts.items()
        ])
        
        # Create relevance prompt
        prompt = f"""
        Determine which of these concepts are still relevant to the current conversation:
        
        QUERY: {query}
        
        RESPONSE: {response}
        
        FORGOTTEN CONCEPTS:
        {concepts_text}
        
        For each concept, return only "relevant" or "not relevant".
        """
        
        try:
            result = self.llm.predict(prompt)
            
            # Parse result
            relevant_concepts = {}
            lines = result.strip().split("\n")
            
            for i, (concept, data) in enumerate(forgotten_concepts.items()):
                if i < len(lines) and "relevant" in lines[i].lower() and "not relevant" not in lines[i].lower():
                    relevant_concepts[concept] = data
                    
            return relevant_concepts
            
        except Exception as e:
            logger.warning(f"Failed to check concept relevance: {e}")
            # Fallback: return all core concepts and a subset of active concepts
            relevant = {}
            for concept, data in forgotten_concepts.items():
                if data.get("priority_tier") == "core" or (data.get("importance", 0) > 0.8 and len(relevant) < 3):
                    relevant[concept] = data
            
            return relevant
            
    def get_stats(self) -> Dict[str, int]:
        """Get recall statistics"""
        return self.recall_stats.copy()
