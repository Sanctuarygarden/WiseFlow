from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import re

logger = logging.getLogger(__name__)

class TreeOfThoughtValidator:
    """
    Implements Tree-of-Thought reasoning for validation
    
    This component:
    1. Explores multiple reasoning paths before making a decision
    2. Evaluates the logical validity of each path
    3. Selects the optimal reasoning approach
    4. Prevents premature convergence on incorrect answers
    """
    
    def __init__(self, llm: Any, max_paths: int = 3, max_depth: int = 2):
        """
        Initialize Tree-of-Thought validator
        
        Args:
            llm: Language model for reasoning
            max_paths: Maximum number of reasoning paths to explore
            max_depth: Maximum depth of reasoning tree
        """
        self.llm = llm
        self.max_paths = max_paths
        self.max_depth = max_depth
        self.stats = {
            "validations_performed": 0,
            "paths_explored": 0,
            "average_paths_per_validation": 0
        }
        
    def validate_response(
        self, 
        query: str, 
        response: str, 
        conversation_history: List[Any]
    ) -> Dict[str, Any]:
        """
        Validate response using Tree-of-Thought reasoning
        
        Args:
            query: User's input query
            response: Generated response to validate
            conversation_history: Previous conversation entries
            
        Returns:
            Validation result with reasoning paths
        """
        self.stats["validations_performed"] += 1
        
        # Step 1: Generate multiple reasoning paths
        reasoning_paths = self._generate_reasoning_paths(query, response, conversation_history)
        self.stats["paths_explored"] += len(reasoning_paths)
        
        # Update average paths per validation
        total_validations = self.stats["validations_performed"]
        total_paths = self.stats["paths_explored"]
        self.stats["average_paths_per_validation"] = total_paths / total_validations if total_validations > 0 else 0
        
        # Step 2: Evaluate each reasoning path
        evaluated_paths = []
        for path in reasoning_paths:
            evaluation = self._evaluate_reasoning_path(path)
            evaluated_paths.append({
                "path": path,
                "evaluation": evaluation
            })
        
        # Step 3: Select the best path
        best_path = max(evaluated_paths, key=lambda p: p["evaluation"]["score"])
        
        # Step 4: Make final validity determination
        is_valid = best_path["evaluation"]["score"] >= 0.7
        issues = []
        
        # Collect issues from best path
        if "issues" in best_path["evaluation"]:
            issues.extend(best_path["evaluation"]["issues"])
        
        # Collect critical issues from all paths
        for path in evaluated_paths:
            if path != best_path and "critical_issues" in path["evaluation"]:
                issues.extend(path["evaluation"]["critical_issues"])
        
        # Deduplicate issues
        unique_issues = []
        for issue in issues:
            if issue not in unique_issues:
                unique_issues.append(issue)
        
        # Calculate total score based on best path
        total_score = int(best_path["evaluation"]["score"] * 50)  # Scale to 0-50
        
        return {
            "is_valid": is_valid,
            "total_score": total_score,
            "confidence": best_path["evaluation"]["score"],
            "issues": unique_issues,
            "reasoning_paths": evaluated_paths,
            "best_path_index": evaluated_paths.index(best_path)
        }
    
    def _generate_reasoning_paths(
        self, 
        query: str, 
        response: str, 
        conversation_history: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate multiple reasoning paths to evaluate the response"""
        # Format conversation history for the prompt
        history_text = self._format_conversation_history(conversation_history)
        
        # Create prompt to generate reasoning paths
        prompt = f"""
        You are evaluating the logical coherence of a response to a query. Generate {self.max_paths} different reasoning paths to analyze this response.
        
        QUERY: {query}
        
        RESPONSE: {response}
        
        CONVERSATION HISTORY:
        {history_text}
        
        For each reasoning path, provide:
        1. A short description of the reasoning approach
        2. The key assumptions being made
        3. Step-by-step reasoning following this approach
        4. Potential flaws in this reasoning path
        
        Generate {self.max_paths} different approaches to analyzing this response. Each approach should consider different aspects or use different reasoning strategies.
        
        Format your response as a JSON array of objects, where each object represents one reasoning path.
        """
        
        try:
            response_text = self.llm.predict(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                reasoning_paths = json.loads(json_match.group(1))
                return reasoning_paths[:self.max_paths]  # Limit to max_paths
                
        except Exception as e:
            logger.warning(f"Failed to generate reasoning paths: {e}")
            
        # Fallback: Generate basic reasoning paths
        return [
            {
                "description": "Direct logical analysis",
                "assumptions": ["The response should directly address the query", "Prior context should be maintained"],
                "reasoning": "Analyze if the response answers the query and maintains contextual continuity",
                "potential_flaws": "May miss nuanced contextual issues"
            },
            {
                "description": "Consistency with conversation history",
                "assumptions": ["The response should not contradict earlier statements", "Critical concepts should be maintained"],
                "reasoning": "Check if the response contradicts any established facts or drops important concepts",
                "potential_flaws": "Might overemphasize historical consistency at expense of new information"
            },
            {
                "description": "Information completeness analysis",
                "assumptions": ["The response should be comprehensive", "All aspects of the query should be addressed"],
                "reasoning": "Evaluate if the response covers all parts of the query adequately",
                "potential_flaws": "May favor verbose responses over concise ones"
            }
        ]
    
    def _evaluate_reasoning_path(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of a reasoning path"""
        # Extract components from path
        description = path.get("description", "")
        assumptions = path.get("assumptions", [])
        reasoning = path.get("reasoning", "")
        potential_flaws = path.get("potential_flaws", "")
        
        # Format for evaluation
        assumptions_text = "\n".join([f"- {a}" for a in assumptions])
        
        # Create prompt for evaluating this path
        prompt = f"""
        Evaluate the strength and validity of this reasoning approach:
        
        APPROACH: {description}
        
        ASSUMPTIONS:
        {assumptions_text}
        
        REASONING PROCESS:
        {reasoning}
        
        POTENTIAL FLAWS:
        {potential_flaws}
        
        Evaluate this reasoning path on:
        1. Logical soundness (0-10)
        2. Comprehensiveness (0-10)
        3. Alignment with standard reasoning norms (0-10)
        
        Then provide:
        - Overall score (0.0-1.0)
        - Main strengths of this reasoning path
        - Critical issues with this reasoning path
        
        Return your evaluation as a JSON object.
        """
        
        try:
            evaluation_text = self.llm.predict(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'(\{.*\})', evaluation_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(1))
                return evaluation
                
        except Exception as e:
            logger.warning(f"Failed to evaluate reasoning path: {e}")
            
        # Fallback evaluation
        return {
            "logical_soundness": 7,
            "comprehensiveness": 7,
            "alignment": 7,
            "score": 0.7,
            "strengths": ["Provides basic logical validation"],
            "critical_issues": [],
            "fallback": True
        }
    
    def _format_conversation_history(self, conversation_history: List[Any]) -> str:
        """Format conversation history for prompts"""
        history_lines = []
        
        for i, entry in enumerate(conversation_history):
            history_lines.append(f"TURN {i+1}:")
            
            # Handle different entry formats
            if hasattr(entry, 'user_input') and hasattr(entry, 'system_response'):
                history_lines.append(f"USER: {entry.user_input}")
                history_lines.append(f"SYSTEM: {entry.system_response}")
            elif isinstance(entry, dict) and 'user_input' in entry and 'system_response' in entry:
                history_lines.append(f"USER: {entry['user_input']}")
                history_lines.append(f"SYSTEM: {entry['system_response']}")
            else:
                # Generic fallback
                history_lines.append(f"ENTRY: {str(entry)}")
                
            history_lines.append("")  # Empty line for separation
        
        return "\n".join(history_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        return self.stats.copy()
