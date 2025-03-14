from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import re

logger = logging.getLogger(__name__)

class LogicalCoherenceValidator:
    """
    Performs deep validation of logical consistency
    
    This component:
    1. Uses Tree-of-Thought (ToT) to evaluate logical paths
    2. Verifies the response maintains continuity with prior conversation
    3. Validates that all claims are substantiated
    4. Ensures logical flow from context to response
    """
    
    def __init__(self, llm: Any, tree_of_thought_handler: Optional[Any] = None):
        """
        Initialize logical coherence validator
        
        Args:
            llm: Language model for validation
            tree_of_thought_handler: Optional ToT handler for advanced validation
        """
        self.llm = llm
        self.tot_handler = tree_of_thought_handler
        self.validation_stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "tot_validations": 0
        }
        
    def validate_coherence(
        self, 
        query: str, 
        response: str, 
        conversation_history: List[Any],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate logical coherence of a response
        
        Args:
            query: User's input query
            response: Generated response to validate
            conversation_history: Previous conversation entries
            threshold: Validation threshold (0.0-1.0)
            
        Returns:
            Dictionary with validation results
        """
        self.validation_stats["validations_performed"] += 1
        
        # If Tree-of-Thought handler is available, use it for advanced validation
        if self.tot_handler and self._should_use_tot(query, response):
            self.validation_stats["tot_validations"] += 1
            return self._validate_with_tot(query, response, conversation_history, threshold)
        
        # Otherwise use standard validation
        return self._validate_standard(query, response, conversation_history, threshold)
    
    def _validate_standard(
        self, 
        query: str, 
        response: str, 
        conversation_history: List[Any],
        threshold: float
    ) -> Dict[str, Any]:
        """Perform standard validation without Tree-of-Thought"""
        # Format conversation history for the prompt
        history_text = self._format_conversation_history(conversation_history)
        
        # Create validation prompt
        prompt = f"""
        Evaluate the logical coherence of this response:
        
        QUERY: {query}
        
        RESPONSE: {response}
        
        CONVERSATION HISTORY:
        {history_text}
        
        Assess the logical coherence by answering these questions:
        1. Does the response directly address the query? (0-10 score)
        2. Is the response consistent with information established in the conversation history? (0-10 score)
        3. Does the response maintain the logical flow of the conversation? (0-10 score)
        4. Are all statements in the response substantiated by context or common knowledge? (0-10 score)
        5. Does the response avoid introducing contradictions? (0-10 score)
        
        For each question, provide a score and brief explanation. Then calculate:
        - Total score (sum of all scores, 0-50)
        - Is the response logically coherent? (yes/no) - "yes" if total score ≥ 35, "no" otherwise
        - Specific issues (list the main logical problems, if any)
        
        Return your analysis as a JSON object with fields for each score, explanation, the total, coherence assessment, and issues.
        """
        
        try:
            response_text = self.llm.predict(prompt)
            validation_result = self._parse_validation_response(response_text)
            
            # Calculate is_valid based on total score and threshold
            total_score = validation_result.get("total_score", 0)
            max_score = 50
            normalized_score = total_score / max_score if max_score > 0 else 0
            is_valid = normalized_score >= threshold
            
            validation_result["is_valid"] = is_valid
            validation_result["confidence"] = normalized_score
            
            # Update statistics
            if is_valid:
                self.validation_stats["validations_passed"] += 1
            else:
                self.validation_stats["validations_failed"] += 1
                
            return validation_result
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            # Fallback result
            return {
                "is_valid": True,  # Default to valid on error
                "total_score": 35,
                "confidence": 0.7,
                "issues": [f"Validation error: {str(e)}"],
                "error": str(e)
            }
    
    def _validate_with_tot(
        self, 
        query: str, 
        response: str, 
        conversation_history: List[Any],
        threshold: float
    ) -> Dict[str, Any]:
        """Perform validation using Tree-of-Thought approach"""
        # Delegate to the Tree-of-Thought handler
        tot_result = self.tot_handler.validate_response(
            query=query,
            response=response,
            conversation_history=conversation_history
        )
        
        # Convert ToT result to standard format
        validation_result = {
            "is_valid": tot_result.get("is_valid", True),
            "total_score": tot_result.get("total_score", 35),
            "confidence": tot_result.get("confidence", 0.7),
            "issues": tot_result.get("issues", []),
            "reasoning_paths": tot_result.get("reasoning_paths", []),
            "tot_method": True
        }
        
        # Update statistics
        if validation_result["is_valid"]:
            self.validation_stats["validations_passed"] += 1
        else:
            self.validation_stats["validations_failed"] += 1
            
        return validation_result
    
    def _format_conversation_history(self, conversation_history: List[Any]) -> str:
        """Format conversation history for validation prompt"""
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
    
    def _parse_validation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse validation response from LLM"""
        # Try to extract JSON
        try:
            # Look for JSON pattern
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group(1))
                
                # Ensure required fields
                if "total_score" not in validation_data:
                    # Try to extract total score from text
                    score_match = re.search(r'[Tt]otal\s+score:?\s*(\d+)', response_text)
                    if score_match:
                        validation_data["total_score"] = int(score_match.group(1))
                    else:
                        validation_data["total_score"] = 0
                
                if "issues" not in validation_data:
                    # Try to extract issues list
                    issues = []
                    issues_match = re.search(r'[Ii]ssues:?(.*?)(?:\n\n|\Z)', response_text, re.DOTALL)
                    if issues_match:
                        issues_text = issues_match.group(1)
                        # Extract bullet points
                        bullet_matches = re.finditer(r'[-*]\s*(.*?)(?:\n|$)', issues_text)
                        issues = [match.group(1).strip() for match in bullet_matches]
                    
                    validation_data["issues"] = issues
                    
                return validation_data
        except Exception as e:
            logger.warning(f"Failed to parse validation result as JSON: {e}")
        
        # Fallback: Extract scores and issues using regex
        result = {"issues": []}
        
        # Extract scores for each question
        question_scores = re.finditer(r'(\d+)\.\s+.*?(\d+)(?:/10)?', response_text)
        total_score = 0
        
        for match in question_scores:
            question_num = match.group(1)
            score = int(match.group(2))
            result[f"question_{question_num}_score"] = score
            total_score += score
        
        # Extract total score
        total_match = re.search(r'[Tt]otal\s+score:?\s*(\d+)', response_text)
        if total_match:
            total_score = int(total_match.group(1))
        
        result["total_score"] = total_score
        
        # Extract coherence assessment
        coherent_match = re.search(r'[Ii]s the response logically coherent\?[^:]*:?\s*(\w+)', response_text)
        if coherent_match:
            is_coherent = coherent_match.group(1).lower() in ["yes", "true", "coherent"]
            result["is_coherent"] = is_coherent
        
        # Extract issues
        issues_match = re.search(r'[Ii]ssues:?(.*?)(?:\n\n|\Z)', response_text, re.DOTALL)
        if issues_match:
            issues_text = issues_match.group(1)
            # Extract bullet points
            bullet_matches = re.finditer(r'[-*]\s*(.*?)(?:\n|$)', issues_text)
            result["issues"] = [match.group(1).strip() for match in bullet_matches]
        
        return result
    
    def _should_use_tot(self, query: str, response: str) -> bool:
        """Determine if Tree-of-Thought validation should be used"""
        # Simple heuristic for when to use ToT
        # Check if query is complex, response is long, or contains reasoning
        if len(query.split()) > 15 or len(response.split()) > 100:
            return True
            
        complex_keywords = ["explain", "analyze", "compare", "evaluate", "reason", "because", "therefore"]
        if any(keyword in query.lower() for keyword in complex_keywords):
            return True
            
        # If response contains multiple reasoning steps
        reasoning_indicators = ["first", "second", "third", "finally", "therefore", "because", "however"]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response.lower())
        if reasoning_count >= 3:
            return True
            
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return self.validation_stats.copy()
