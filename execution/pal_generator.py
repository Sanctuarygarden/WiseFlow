from typing import Dict, Any, List, Optional, Tuple
import logging
import tempfile
import os
import subprocess
import re
import time
import json

logger = logging.getLogger(__name__)

class PALEnhancedCodeGenerator:
    """
    Enhances code generation with Program-Aided Language model capabilities
    
    This component:
    1. Generates partial implementations to test hypotheses
    2. Executes generated code to verify correctness
    3. Refines based on execution results
    4. Tracks execution history to avoid repeating failed approaches
    """
    
    def __init__(
        self, 
        llm: Any, 
        code_analyzer: Any, 
        max_iterations: int = 3,
        execution_timeout: int = 5  # Seconds
    ):
        """
        Initialize PAL-enhanced code generator
        
        Args:
            llm: Language model for code generation
            code_analyzer: Code analyzer for structure analysis
            max_iterations: Maximum refinement iterations
            execution_timeout: Timeout for code execution in seconds
        """
        self.llm = llm
        self.analyzer = code_analyzer
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.execution_history = []
        self.generation_stats = {
            "total_generations": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "refinement_cycles": 0
        }
        
    def generate_code(
        self,
        language: str,
        purpose: str,
        dependencies: Optional[List[Any]] = None,
        existing_code: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        style_preferences: Optional[Dict[str, Any]] = None,
        structure_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate code with execution-based validation
        
        Args:
            language: Programming language to generate
            purpose: Description of what the code should do
            dependencies: Related code blocks to maintain compatibility with
            existing_code: Existing code to modify (if any)
            requirements: Specific requirements the code must meet
            style_preferences: Code style preferences
            structure_preferences: Code structure preferences
            
        Returns:
            Dictionary with generated code and metadata
        """
        self.generation_stats["total_generations"] += 1
        
        # If language not supported by PAL or not appropriate for PAL, use basic generation
        if language not in ["python", "javascript"] or not self._should_use_pal(purpose, requirements):
            return self._generate_basic_code(
                language, purpose, dependencies, existing_code, 
                requirements, style_preferences, structure_preferences
            )
        
        # PAL approach: Generate + Execute + Refine
        iterations = 0
        code = None
        execution_results = []
        refinement_history = []
        
        # Initial code generation
        initial_result = self._generate_basic_code(
            language, purpose, dependencies, existing_code, 
            requirements, style_preferences, structure_preferences
        )
        code = initial_result["code"]
        
        while iterations < self.max_iterations:
            # Create test harness
            test_code = self._create_test_harness(code, language, purpose, requirements)
            
            # Execute code
            execution_result = self._safely_execute_code(test_code, language)
            execution_results.append(execution_result)
            
            # Record execution history
            execution_entry = {
                "iteration": iterations,
                "code": code,
                "test_code": test_code,
                "result": execution_result,
                "timestamp": time.time()
            }
            self.execution_history.append(execution_entry)
            
            # Check if execution succeeded
            if execution_result["success"]:
                self.generation_stats["successful_executions"] += 1
                # Return successful code with execution result
                return {
                    "code": code,
                    "language": language,
                    "purpose": purpose,
                    "analysis": self.analyzer.analyze_block(code, language),
                    "execution_result": execution_result,
                    "iterations": iterations + 1,
                    "execution_history": execution_results
                }
            
            self.generation_stats["failed_executions"] += 1
            
            # Refine code based on error
            iterations += 1
            refinement = self._refine_based_on_error(
                code=code,
                language=language,
                error=execution_result["error"],
                output=execution_result.get("output", ""),
                purpose=purpose,
                dependencies=dependencies,
                requirements=requirements
            )
            
            refinement_history.append(refinement)
            code = refinement["code"]
            
            self.generation_stats["refinement_cycles"] += 1
        
        # If we reach here, all iterations failed
        # Return the last version with execution history
        return {
            "code": code,
            "language": language,
            "purpose": purpose,
            "analysis": self.analyzer.analyze_block(code, language),
            "execution_result": execution_results[-1] if execution_results else {"success": False, "error": "Maximum iterations reached"},
            "iterations": iterations,
            "execution_history": execution_results,
            "refinement_history": refinement_history,
            "warning": "Maximum iterations reached without successful execution"
        }
    
    def _generate_basic_code(
        self,
        language: str,
        purpose: str,
        dependencies: Optional[List[Any]] = None,
        existing_code: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        style_preferences: Optional[Dict[str, Any]] = None,
        structure_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate code without execution validation"""
        # Format dependencies for context
        dependencies_text = ""
        if dependencies:
            dependencies_text = "DEPENDENT MODULES:\n"
            for i, dep in enumerate(dependencies):
                dependencies_text += f"Module {i+1} ({dep.file_path or 'unnamed'}):\n"
                dependencies_text += f"```{dep.language}\n{dep.content}\n```\n\n"
        
        # Format requirements
        requirements_text = ""
        if requirements:
            requirements_text = "SPECIFIC REQUIREMENTS:\n"
            requirements_text += "\n".join([f"- {req}" for req in requirements])
        
        # Format style preferences
        style_text = ""
        if style_preferences:
            style_text = "STYLE PREFERENCES:\n"
            style_text += "\n".join([f"- {k}: {v}" for k, v in style_preferences.items()])
        
        # Format structure preferences
        structure_text = ""
        if structure_preferences:
            structure_text = "STRUCTURE PREFERENCES:\n"
            structure_text += "\n".join([f"- {k}: {v}" for k, v in structure_preferences.items()])
        
        # Create generation prompt
        operation = "modify" if existing_code else "generate"
        prompt = f"""
        You are a skilled programmer. I need you to {operation} {language} code for the following purpose:
        
        PURPOSE:
        {purpose}
        
        {dependencies_text}
        
        {requirements_text}
        
        {style_text}
        
        {structure_text}
        
        {"EXISTING CODE TO MODIFY:" if existing_code else ""}
        {f"```{language}\n{existing_code}\n```" if existing_code else ""}
        
        Please provide:
        1. Clean, well-structured {language} code that fulfills the purpose
        2. Good error handling and comments
        3. Compatibility with any dependent modules shown
        
        Return just the code itself without explanation.
        """
        
        generated_code = self.llm.predict(prompt)
        
        # Clean up response to extract just the code
        code = self._extract_code_from_response(generated_code, language)
        
        # Analyze the generated code
        analysis = self.analyzer.analyze_block(code, language)
        
        return {
            "code": code,
            "language": language,
            "purpose": purpose,
            "analysis": analysis
        }
    
    def _should_use_pal(self, purpose: str, requirements: Optional[List[str]] = None) -> bool:
        """Determine if PAL approach should be used based on complexity"""
        # Use simple heuristics to determine if execution verification would be valuable
        purpose_lower = purpose.lower() if purpose else ""
        
        # PAL is valuable for algorithmic or computation tasks
        if any(kw in purpose_lower for kw in ["algorithm", "function", "calculation", "process", "compute"]):
            return True
            
        # PAL is valuable when testing is mentioned in requirements
        if requirements and any("test" in req.lower() for req in requirements):
            return True
            
        # PAL is valuable for data processing or transformation
        if any(kw in purpose_lower for kw in ["parse", "transform", "convert", "process data", "analyze data"]):
            return True
            
        # PAL is valuable for validation logic
        if any(kw in purpose_lower for kw in ["validate", "verify", "check", "ensure"]):
            return True
            
        return False
    
    def _create_test_harness(
        self, 
        code: str, 
        language: str, 
        purpose: str, 
        requirements: Optional[List[str]] = None
    ) -> str:
        """Create a minimal test harness for the generated code"""
        prompt = f"""
        Create a minimal test harness for this {language} code:
        
        ```{language}
        {code}
        ```
        
        PURPOSE: {purpose}
        
        {"REQUIREMENTS:" if requirements else ""}
        {chr(10).join([f"- {req}" for req in requirements]) if requirements else ""}
        
        The test harness should:
        1. Import the necessary dependencies
        2. Create sample inputs
        3. Call the main functionality
        4. Print or display the results
        
        Keep it minimal but complete enough to verify the code works correctly.
        Return only the code for the test harness.
        """
        
        test_harness = self.llm.predict(prompt)
        
        # Extract code from the response
        test_code = self._extract_code_from_response(test_harness, language)
        
        # For Python, we can combine the original code and test
        if language == "python":
            return f"{code}\n\n# Test harness\n{test_code}"
        
        # For JavaScript, we might need to adjust based on module system
        if language == "javascript":
            # Simple check if code uses ES modules
            if "export " in code or "import " in code:
                # Create a temporary module system
                return f"""
                // Original code as module
                const moduleCode = `{code.replace('`', '\\`')}`;
                
                // Write to temporary file
                const fs = require('fs');
                fs.writeFileSync('temp_module.js', moduleCode);
                
                // Test harness
                {test_code}
                
                // Clean up
                fs.unlinkSync('temp_module.js');
                """
            else:
                # Simple concatenation for non-module code
                return f"{code}\n\n// Test harness\n{test_code}"
        
        # Default fallback
        return f"{code}\n\n// Test harness\n{test_code}"
    
    def _safely_execute_code(self, code: str, language: str) -> Dict[str, Any]:
        """Safely execute code in a sandbox environment"""
        if language == "python":
            return self._execute_python_code(code)
        elif language == "javascript":
            return self._execute_javascript_code(code)
        else:
            return {
                "success": False,
                "error": f"Execution not supported for {language}"
            }
    
    def _execute_python_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a safe environment"""
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                f.write(code.encode())
                temp_file = f.name
            
            # Execute with timeout
            try:
                result = subprocess.run(
                    ["python", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout
                )
                
                # Clean up
                os.unlink(temp_file)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": result.stdout,
                        "stderr": result.stderr
                    }
                else:
                    return {
                        "success": False,
                        "error": result.stderr,
                        "output": result.stdout
                    }
            except subprocess.TimeoutExpired:
                # Clean up
                os.unlink(temp_file)
                return {
                    "success": False,
                    "error": f"Execution timed out after {self.execution_timeout} seconds"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_javascript_code(self, code: str) -> Dict[str, Any]:
        """Execute JavaScript code in a safe environment"""
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
                f.write(code.encode())
                temp_file = f.name
            
            # Execute with timeout
            try:
                result = subprocess.run(
                    ["node", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout
                )
                
                # Clean up
                os.unlink(temp_file)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": result.stdout,
                        "stderr": result.stderr
                    }
                else:
                    return {
                        "success": False,
                        "error": result.stderr,
                        "output": result.stdout
                    }
            except subprocess.TimeoutExpired:
                # Clean up
                os.unlink(temp_file)
                return {
                    "success": False,
                    "error": f"Execution timed out after {self.execution_timeout} seconds"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _refine_based_on_error(
        self,
        code: str,
        language: str,
        error: str,
        output: str,
        purpose: str,
        dependencies: Optional[List[Any]] = None,
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Refine code based on execution error"""
        # Check if this is an error we've seen before
        similar_errors = self._find_similar_errors(error)
        previous_attempts = [entry["code"] for entry in similar_errors]
        
        # Create a refinement prompt
        prompt = f"""
        The following {language} code failed with this error:
        
        ```{language}
        {code}
        ```
        
        ERROR:
        {error}
        
        {"PARTIAL OUTPUT:" if output else ""}
        {output if output else ""}
        
        PURPOSE:
        {purpose}
        
        {"REQUIREMENTS:" if requirements else ""}
        {chr(10).join([f"- {req}" for req in requirements]) if requirements else ""}
        
        {"PREVIOUSLY ATTEMPTED SOLUTIONS THAT FAILED:" if previous_attempts else ""}
        {chr(10).join([f"```{language}\n{attempt}\n```" for attempt in previous_attempts]) if previous_attempts else ""}
        
        Please fix the code to address this error. Focus on:
        1. Addressing the specific error message
        2. Ensuring the code fulfills its purpose
        3. Making sure the code is robust and handles edge cases
        {"4. Avoiding approaches from previous failed attempts" if previous_attempts else ""}
        
        Return only the fixed code.
        """
        
        fixed_code = self.llm.predict(prompt)
        
        # Extract code from the response
        fixed_code_clean = self._extract_code_from_response(fixed_code, language)
        
        # Run analysis on the fixed code
        analysis = self.analyzer.analyze_block(fixed_code_clean, language)
        
        return {
            "code": fixed_code_clean,
            "language": language,
            "purpose": purpose,
            "analysis": analysis,
            "previous_error": error
        }
    
    def _find_similar_errors(self, error: str) -> List[Dict[str, Any]]:
        """Find similar errors in execution history"""
        similar_errors = []
        
        for entry in self.execution_history:
            if "result" in entry and "error" in entry["result"]:
                entry_error = entry["result"]["error"]
                
                # Calculate similarity
                similarity = self._calculate_error_similarity(error, entry_error)
                
                if similarity > 0.7:  # Arbitrary threshold
                    similar_errors.append(entry)
        
        return similar_errors
    
    def _calculate_error_similarity(self, error1: str, error2: str) -> float:
        """Calculate similarity between two error messages"""
        # Simple token-based similarity
        if not error1 or not error2:
            return 0.0
            
        tokens1 = set(error1.lower().split())
        tokens2 = set(error2.lower().split())
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code block from LLM response"""
        # Look for code blocks with language marker
        language_pattern = f"```{language}(.*?)```"
        language_matches = re.findall(language_pattern, response, re.DOTALL)
        
        if language_matches:
            return language
