from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConceptReevaluationSystem:
    """
    Periodically re-evaluates concepts to ensure they are still relevant
    
    This component:
    1. Tracks when concepts were last actively used or referenced
    2. Periodically checks if dropped concepts are still relevant to overall context
    3. Explicitly confirms concept status before allowing it to fade
    4. Prevents unintentional concept loss through neglect
    """
    
    def __init__(self, context_tracker: Any, llm: Optional[Any] = None):
        """
        Initialize concept reevaluation system
        
        Args:
            context_tracker: Context tracker for accessing concept registry
            llm: Optional language model for relevance assessment
        """
        self.context_tracker = context_tracker
        self.llm = llm
        self.last_reevaluation = {}  # concept -> last_reevaluation_timestamp
        self.reevaluation_results = {}  # concept -> [reevaluation_results]
        self.reevaluation_stats = {
            "total_reevaluations": 0,
            "concepts_retained": 0,
            "concepts_deprecated": 0
        }
        
    def should_reevaluate(self, concept: str, threshold_days: int = 7) -> bool:
        """
        Determine if a concept should be reevaluated
        
        Args:
            concept: Concept name to check
            threshold_days: Days since last mention to trigger reevaluation
            
        Returns:
            Boolean indicating if reevaluation is needed
        """
        # Skip if concept not in registry
        if concept not in self.context_tracker.concept_registry:
            return False
        
        concept_data = self.context_tracker.concept_registry[concept]
        current_time = datetime.now()
        
        # Check when concept was last mentioned
        last_mentioned = None
        if "last_mentioned" in concept_data:
            try:
                last_mentioned = datetime.fromisoformat(concept_data["last_mentioned"])
            except (ValueError, TypeError):
                pass
        
        # Check when concept was last updated
        last_updated = None
        if "last_updated" in concept_data:
            try:
                last_updated = datetime.fromisoformat(concept_data["last_updated"])
            except (ValueError, TypeError):
                pass
        
        # Get the most recent of last_mentioned and last_updated
        last_activity = None
        if last_mentioned and last_updated:
            last_activity = max(last_mentioned, last_updated)
        elif last_mentioned:
            last_activity = last_mentioned
        elif last_updated:
            last_activity = last_updated
        
        # If no activity timestamp found, use today minus threshold as fallback
        if not last_activity:
            return True
        
        # Check if concept has been inactive for threshold period
        inactive_period = current_time - last_activity
        if inactive_period > timedelta(days=threshold_days):
            # Now check when it was last reevaluated
            if concept in self.last_reevaluation:
                try:
                    last_reeval = datetime.fromisoformat(self.last_reevaluation[concept])
                    # Only reevaluate if it's been at least threshold_days/2 since last reevaluation
                    return (current_time - last_reeval) > timedelta(days=threshold_days/2)
                except (ValueError, TypeError):
                    return True
            return True
        
        return False
    
    def find_concepts_needing_reevaluation(
        self, 
        importance_threshold: float = 0.6,
        max_count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find concepts that need reevaluation
        
        Args:
            importance_threshold: Minimum importance score to consider
            max_count: Maximum number of concepts to return
            
        Returns:
            List of concepts needing reevaluation with their data
        """
        candidates = []
        
        # Check all concepts in registry
        for concept, data in self.context_tracker.concept_registry.items():
            # Skip low importance concepts
            importance = data.get("importance", 0.0)
            if importance < importance_threshold:
                continue
                
            # Check if reevaluation is needed
            if self.should_reevaluate(concept):
                candidates.append({
                    "concept": concept,
                    "data": data.copy(),
                    "importance": importance
                })
        
        # Sort by importance (highest first)
        candidates.sort(key=lambda x: x["importance"], reverse=True)
        
        # Return top N candidates
        return candidates[:max_count]
    
    def reevaluate_concept(
        self, 
        concept: str, 
        recent_entries: List[Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reevaluate a concept to determine if it's still relevant
        
        Args:
            concept: Concept to reevaluate
            recent_entries: Recent conversation entries for context
            project_vision: Optional project vision for long-term relevance
            
        Returns:
            Reevaluation result
        """
        self.reevaluation_stats["total_reevaluations"] += 1
        current_time = datetime.now().isoformat()
        
        # Get concept data
        if concept not in self.context_tracker.concept_registry:
            return {
                "concept": concept,
                "status": "not_found",
                "still_relevant": False,
                "timestamp": current_time
            }
            
        concept_data = self.context_tracker.concept_registry[concept]
        definition = concept_data.get("definition", "No definition available")
        importance = concept_data.get("importance", 0.0)
        tier = concept_data.get("priority_tier", "unknown")
        
        # If this is a CORE concept, always consider it relevant
        if tier == "core":
            self.reevaluation_stats["concepts_retained"] += 1
            
            # Update last reevaluation timestamp
            self.last_reevaluation[concept] = current_time
            
            return {
                "concept": concept,
                "status": "retained",
                "reason": "core_concept",
                "still_relevant": True,
                "timestamp": current_time
            }
        
        # If no LLM available, use simple heuristic
        if not self.llm:
            # Keep high importance concepts, deprecate others
            still_relevant = importance >= 0.7
            
            if still_relevant:
                self.reevaluation_stats["concepts_retained"] += 1
            else:
                self.reevaluation_stats["concepts_deprecated"] += 1
                
            # Update last reevaluation timestamp
            self.last_reevaluation[concept] = current_time
            
            return {
                "concept": concept,
                "status": "retained" if still_relevant else "deprecated",
                "reason": "importance_heuristic",
                "still_relevant": still_relevant,
                "timestamp": current_time
            }
        
        # Format recent entries for the prompt
        entries_text = self._format_recent_entries(recent_entries)
        
        # Format project vision for the prompt
        vision_text = "No project vision available."
        if project_vision:
            vision_parts = []
            for key, value in project_vision.items():
                if isinstance(value, str):
                    vision_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    vision_parts.append(f"{key}: {', '.join(value)}")
            vision_text = "\n".join(vision_parts)
        
        # Create reevaluation prompt
        prompt = f"""
        Evaluate whether this concept is still relevant to the current conversation and project:
        
        CONCEPT: {concept}
        
        DEFINITION: {definition}
        
        IMPORTANCE: {importance:.2f} out of 1.0
        
        PRIORITY TIER: {tier}
        
        RECENT CONVERSATION:
        {entries_text}
        
        PROJECT VISION:
        {vision_text}
        
        Determining if this concept:
        1. Is still directly relevant to the current conversation topics
        2. Is fundamental to the project's goals and vision
        3. Should be retained even if not recently mentioned
        4. Should be deprecated (marked as no longer relevant)
        
        Return your assessment as a JSON object with fields:
        - still_relevant (true/false)
        - confidence (0.0-1.0)
        - reasoning (string explanation)
        - action (either "retain" or "deprecate")
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Parse JSON response
            import re
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                assessment = json.loads(json_match.group(0))
                
                # Get the main decision
                still_relevant = assessment.get("still_relevant", True)
                
                if still_relevant:
                    self.reevaluation_stats["concepts_retained"] += 1
                else:
                    self.reevaluation_stats["concepts_deprecated"] += 1
                
                # Update last reevaluation timestamp
                self.last_reevaluation[concept] = current_time
                
                # Store reevaluation result
                result = {
                    "concept": concept,
                    "status": "retained" if still_relevant else "deprecated",
                    "reason": assessment.get("reasoning", "LLM assessment"),
                    "confidence": assessment.get("confidence", 0.7),
                    "still_relevant": still_relevant,
                    "timestamp": current_time
                }
                
                # Update reevaluation history
                if concept not in self.reevaluation_results:
                    self.reevaluation_results[concept] = []
                self.reevaluation_results[concept].append(result)
                
                return result
                
        except Exception as e:
            logger.warning(f"Concept reevaluation failed: {e}")
            
        # Fallback: retain the concept
        self.reevaluation_stats["concepts_retained"] += 1
        
        # Update last reevaluation timestamp
        self.last_reevaluation[concept] = current_time
        
        return {
            "concept": concept,
            "status": "retained",
            "reason": "reevaluation_error_fallback",
            "still_relevant": True,
            "timestamp": current_time
        }
    
    def deprecate_concept(self, concept: str, reason: str) -> bool:
        """
        Mark a concept as deprecated
        
        Args:
            concept: Concept to deprecate
            reason: Reason for deprecation
            
        Returns:
            Success flag
        """
        if concept not in self.context_tracker.concept_registry:
            return False
            
        # Update concept data in registry
        concept_data = self.context_tracker.concept_registry[concept]
        
        # Add deprecation marker
        concept_data["deprecated"] = True
        concept_data["deprecation_reason"] = reason
        concept_data["deprecation_timestamp"] = datetime.now().isoformat()
        
        # Lower importance to ensure it's not prioritized
        concept_data["importance"] = min(0.3, concept_data.get("importance", 0.5))
        
        # Update priority tier
        concept_data["priority_tier"] = "background"
        
        # Update registry
        self.context_tracker.concept_registry[concept] = concept_data
        
        return True
    
    def run_reevaluation_cycle(
        self, 
        recent_entries: List[Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete reevaluation cycle for all eligible concepts
        
        Args:
            recent_entries: Recent conversation entries for context
            project_vision: Optional project vision for long-term relevance
            
        Returns:
            Cycle results
        """
        # Find concepts needing reevaluation
        candidates = self.find_concepts_needing_reevaluation()
        
        if not candidates:
            return {
                "status": "no_candidates",
                "timestamp": datetime.now().isoformat(),
                "concepts_processed": 0
            }
        
        # Process each candidate
        results = []
        
        for candidate in candidates:
            concept = candidate["concept"]
            
            # Reevaluate the concept
            result = self.reevaluate_concept(
                concept=concept,
                recent_entries=recent_entries,
                project_vision=project_vision
            )
            
            # Apply deprecation if needed
            if not result["still_relevant"]:
                self.deprecate_concept(
                    concept=concept,
                    reason=result.get("reason", "Reevaluation determined concept is no longer relevant")
                )
            
            results.append(result)
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "concepts_processed": len(results),
            "results": results
        }
    
    def _format_recent_entries(self, entries: List[Any]) -> str:
        """Format recent entries for prompts"""
        formatted = []
        
        for i, entry in enumerate(entries):
            formatted.append(f"ENTRY {i+1}:")
            
            # Handle different entry formats
            if hasattr(entry, 'user_input') and hasattr(entry, 'system_response'):
                formatted.append(f"USER: {entry.user_input}")
                formatted.append(f"SYSTEM: {entry.system_response}")
            elif isinstance(entry, dict) and 'user_input' in entry and 'system_response' in entry:
                formatted.append(f"USER: {entry['user_input']}")
                formatted.append(f"SYSTEM: {entry['system_response']}")
            else:
                # Generic fallback
                formatted.append(f"ENTRY: {str(entry)}")
                
            formatted.append("")  # Empty line for separation
        
        return "\n".join(formatted)
    
    def get_stats(self) -> Dict[str, int]:
        """Get reevaluation statistics"""
        return self.reevaluation_stats.copy()

# Factory function for creating the reevaluation system
def create_reevaluation_system(context_tracker: Any, llm: Any) -> ConceptReevaluationSystem:
    """Create a fully initialized concept reevaluation system"""
    return ConceptReevaluationSystem(
        context_tracker=context_tracker,
        llm=llm
    )

