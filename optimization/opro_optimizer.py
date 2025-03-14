from typing import Dict, Any, List, Optional, Callable
import logging
import json
import time
import random
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class OPROPromptOptimizer:
    """
    Self-improving prompt optimization using OPRO technique
    
    This component:
    1. Tracks prompt performance over time
    2. Generates improved versions of prompts
    3. A/B tests prompt variants
    4. Handles prompt regression testing
    """
    
    def __init__(self, llm: Any):
        """
        Initialize OPRO prompt optimizer
        
        Args:
            llm: Language model for optimization
        """
        self.llm = llm
        self.prompt_registry = {}  # prompt_id -> {versions: [...], performance: [...]}
        self.optimization_stats = {
            "prompts_registered": 0,
            "optimizations_performed": 0,
            "regressions_detected": 0,
            "improvements_found": 0
        }
        
    def register_prompt(
        self, 
        prompt_id: str, 
        initial_prompt: str, 
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a prompt for optimization
        
        Args:
            prompt_id: Unique identifier for the prompt
            initial_prompt: Initial prompt text
            description: Description of prompt purpose
            metadata: Optional metadata for the prompt
            
        Returns:
            Success flag
        """
        if prompt_id in self.prompt_registry:
            logger.warning(f"Prompt ID {prompt_id} is already registered")
            return False
            
        self.prompt_registry[prompt_id] = {
            "versions": [{
                "prompt": initial_prompt,
                "version": 1,
                "created_at": datetime.now().isoformat()
            }],
            "current_version": 1,
            "description": description,
            "metadata": metadata or {},
            "performance": [],
            "optimization_count": 0
        }
        
        self.optimization_stats["prompts_registered"] += 1
        logger.info(f"Registered prompt {prompt_id}")
        return True
        
    def get_current_prompt(self, prompt_id: str) -> Optional[str]:
        """
        Get the current best version of a prompt
        
        Args:
            prompt_id: ID of prompt to retrieve
            
        Returns:
            Current prompt text, or None if not found
        """
        if prompt_id not in self.prompt_registry:
            logger.warning(f"Prompt ID {prompt_id} not found")
            return None
            
        registry = self.prompt_registry[prompt_id]
        current_version = registry["current_version"]
        
        for version in registry["versions"]:
            if version["version"] == current_version:
                return version["prompt"]
                
        # Fallback to the latest version if current_version not found
        return registry["versions"][-1]["prompt"]
        
    def record_performance(
        self, 
        prompt_id: str, 
        success_score: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record prompt performance for future optimization
        
        Args:
            prompt_id: ID of prompt to record performance for
            success_score: Score indicating performance (0.0-1.0)
            metadata: Optional metadata about the usage context
            
        Returns:
            Success flag
        """
        if prompt_id not in self.prompt_registry:
            logger.warning(f"Cannot record performance for unknown prompt ID {prompt_id}")
            return False
            
        registry = self.prompt_registry[prompt_id]
        current_version = registry["current_version"]
        
        registry["performance"].append({
            "version": current_version,
            "success_score": success_score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Check if we should optimize (every 10 performance entries)
        if len(registry["performance"]) % 10 == 0:
            self.optimize_prompt(prompt_id)
            
        return True
        
    def optimize_prompt(self, prompt_id: str) -> Optional[str]:
        """
        Generate an improved version of the prompt based on performance data
        
        Args:
            prompt_id: ID of prompt to optimize
            
        Returns:
            Improved prompt text, or None if optimization failed
        """
        if prompt_id not in self.prompt_registry:
            logger.warning(f"Cannot optimize unknown prompt ID {prompt_id}")
            return None
            
        registry = self.prompt_registry[prompt_id]
        
        # Need at least 5 performance entries to optimize
        if len(registry["performance"]) < 5:
            logger.info(f"Not enough performance data to optimize prompt {prompt_id}")
            return None
            
        # Get current prompt
        current_prompt = self.get_current_prompt(prompt_id)
        current_version = registry["current_version"]
        
        if not current_prompt:
            logger.warning(f"Failed to get current prompt for {prompt_id}")
            return None
            
        # Get recent performance data
        recent_performance = registry["performance"][-10:]
        
        # Calculate average success score
        avg_score = sum(p["success_score"] for p in recent_performance) / len(recent_performance)
        
        # If performance is very good (>0.9), don't optimize
        if avg_score > 0.9:
            logger.info(f"Prompt {prompt_id} performance is excellent ({avg_score:.2f}), no optimization needed")
            return None
            
        # Extract successful and unsuccessful examples
        successful = [p for p in recent_performance if p["success_score"] > 0.7]
        unsuccessful = [p for p in recent_performance if p["success_score"] < 0.5]
        
        # Need at least some examples of each
        if not successful or not unsuccessful:
            logger.info(f"Not enough diverse performance data to optimize prompt {prompt_id}")
            return None
            
        # Format examples for the optimization prompt
        successful_examples = "\n".join([
            f"Example {i+1}: {s.get('metadata', {}).get('query', 'Unknown query')}"
            for i, s in enumerate(successful[:3])  # Just use top 3
        ])
        
        unsuccessful_examples = "\n".join([
            f"Example {i+1}: {s.get('metadata', {}).get('query', 'Unknown query')}"
            for i, s in enumerate(unsuccessful[:3])  # Just use top 3
        ])
        
        # Create optimization prompt
        prompt = f"""
        You are an expert prompt engineer. Your task is to improve this prompt:
        
        CURRENT PROMPT:
        {current_prompt}
        
        PROMPT PURPOSE:
        {registry["description"]}
        
        The prompt's current average performance is {avg_score:.2f} out of 1.0.
        
        SUCCESSFUL EXAMPLES where the prompt worked well:
        {successful_examples}
        
        UNSUCCESSFUL EXAMPLES where the prompt performed poorly:
        {unsuccessful_examples}
        
        Please improve the prompt by:
        1. Making it more specific and clear
        2. Addressing failure patterns in the unsuccessful examples
        3. Preserving elements that work well in successful examples
        4. Reducing ambiguity and improving structure
        
        Return only the improved prompt, with no explanations.
        """
        
        try:
            improved_prompt = self.llm.predict(prompt)
            
            # Register new version
            new_version = current_version + 1
            registry["versions"].append({
                "prompt": improved_prompt,
                "version": new_version,
                "created_at": datetime.now().isoformat(),
                "previous_version": current_version,
                "optimization_round": registry["optimization_count"] + 1
            })
            
            registry["current_version"] = new_version
            registry["optimization_count"] += 1
            
            self.optimization_stats["optimizations_performed"] += 1
            logger.info(f"Optimized prompt {prompt_id} from version {current_version} to {new_version}")
            
            return improved_prompt
            
        except Exception as e:
            logger.warning(f"Failed to optimize prompt {prompt_id}: {e}")
            return None
            
    def run_regression_test(
        self, 
        prompt_id: str, 
        test_cases: List[Dict[str, Any]],
        test_function: Optional[Callable] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run regression tests on prompt versions to validate improvements
        
        Args:
            prompt_id: Prompt ID to test
            test_cases: List of test cases {input, expected_output}
            test_function: Optional custom test function
            
        Returns:
            Test results, or None if testing failed
        """
        if prompt_id not in self.prompt_registry:
            logger.warning(f"Cannot test unknown prompt ID {prompt_id}")
            return None
            
        registry = self.prompt_registry[prompt_id]
        
        # Get top 2 latest versions to compare
        versions = sorted(registry["versions"], key=lambda v: v["version"], reverse=True)
        
        if len(versions) < 2:
            logger.info(f"Not enough versions of prompt {prompt_id} for regression testing")
            return {
                "status": "insufficient_versions",
                "prompt_id": prompt_id
            }
            
        latest = versions[0]
        previous = versions[1]
        
        latest_results = []
        previous_results = []
        
        # Test each prompt version on the test cases
        for test_case in test_cases:
            # Use custom test function if provided
            if test_function:
                latest_result = test_function(latest["prompt"], test_case)
                previous_result = test_function(previous["prompt"], test_case)
            else:
                # Default testing function
                latest_result = self._test_prompt(latest["prompt"], test_case)
                previous_result = self._test_prompt(previous["prompt"], test_case)
            
            latest_results.append(latest_result)
            previous_results.append(previous_result)
        
        # Calculate scores
        latest_score = sum(r["score"] for r in latest_results) / len(latest_results)
        previous_score = sum(r["score"] for r in previous_results) / len(previous_results)
        
        # Compare versions
        improvement = latest_score - previous_score
        
        if latest_score >= previous_score:
            status = "improvement"
            if improvement > 0.1:
                self.optimization_stats["improvements_found"] += 1
        else:
            # If regression, revert to previous version
            registry["current_version"] = previous["version"]
            status = "regression_reverted"
            self.optimization_stats["regressions_detected"] += 1
            logger.warning(f"Regression detected for prompt {prompt_id}, reverting to version {previous['version']}")
            
        return {
            "status": status,
            "prompt_id": prompt_id,
            "latest_version": latest["version"],
            "previous_version": previous["version"],
            "latest_score": latest_score,
            "previous_score": previous_score,
            "improvement": improvement,
            "test_count": len(test_cases),
            "latest_results": latest_results,
            "previous_results": previous_results
        }
        
    def _test_prompt(
        self, 
        prompt: str, 
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a prompt with given input and expected output
        
        Args:
            prompt: Prompt to test
            test_case: Test case with input and expected output
            
        Returns:
            Test result with score
        """
        input_text = test_case.get("input", "")
        expected_output = test_case.get("expected_output", "")
        
        if not input_text or not expected_output:
            return {
                "score": 0.5,
                "reason": "Invalid test case"
            }
        
        try:
            # Fill the prompt template with the input
            filled_prompt = prompt.replace("{input}", input_text)
            
            # Send to the LLM
            actual_output = self.llm.predict(filled_prompt)
            
            # Compare output to expected
            similarity_score = self._calculate_output_similarity(actual_output, expected_output)
            
            return {
                "score": similarity_score,
                "input": input_text,
                "expected": expected_output,
                "actual": actual_output
            }
            
        except Exception as e:
            logger.warning(f"Error testing prompt: {e}")
            
            # Return a low score on error
            return {
                "score": 0.1,
                "error": str(e),
                "input": input_text
            }
    
    def _calculate_output_similarity(self, output1: str, output2: str) -> float:
        """
        Calculate similarity between LLM outputs
        
        Args:
            output1: First output
            output2: Second output
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple token-based similarity
        if not output1 or not output2:
            return 0.0
            
        tokens1 = set(output1.lower().split())
        tokens2 = set(output2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def get_prompt_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get history of a prompt's versions and performance
        
        Args:
            prompt_id: Prompt ID to get history for
            
        Returns:
            Prompt history, or None if not found
        """
        if prompt_id not in self.prompt_registry:
            logger.warning(f"Prompt ID {prompt_id} not found")
            return None
            
        return self.prompt_registry[prompt_id].copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get optimization statistics"""
        return self.optimization_stats.copy()
