import os
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

# Import core components
from core.meta_controller import MetaControllerAgent
from core.event_bus import EventBus
from core.feature_config import FeatureConfig

# Import context management components
from context.consolidation import ContextConsolidationLayer
from context.evolution_detector import ProactiveConceptEvolutionDetector
from context.recall_system import ConceptRecallSystem

# Import validation components
from validation.coherence_validator import LogicalCoherenceValidator
from validation.tree_of_thought import TreeOfThoughtValidator
from validation.lazy_validator import LazyValidator

# Import execution components
from execution.feedback_modifier import FeedbackBasedExecutionModifier
from execution.pal_generator import PALEnhancedCodeGenerator
from execution.graph_analyzer import GraphOfThoughtCodeAnalyzer

# Import optimization components
from optimization.opro_optimizer import OPROPromptOptimizer
from optimization.quality_monitor import QualityMonitor

# Configure logging
logger = logging.getLogger(__name__)

class IntegratedMetaControllerAgent(MetaControllerAgent):
    """
    Enhanced Meta-Controller with full integration of all specialized components
    
    This agent:
    1. Orchestrates all specialized agents and components
    2. Ensures comprehensive context awareness and continuity
    3. Maintains logical consistency through multi-layer validation
    4. Adapts to user feedback in real-time
    5. Employs advanced reasoning techniques for complex queries
    """
    
    def __init__(
        self,
        memory_agent: Any,
        context_agent: Any,
        evolution_agent: Any,
        execution_agent: Any,
        context_tracker: Any,
        llm: Optional[Any] = None,
        temperature: float = 0.2,
        prompts: Dict[str, str] = None,
        config_path: Optional[str] = None
    ):
        # Initialize base MetaControllerAgent
        super().__init__(
            memory_agent=memory_agent,
            context_agent=context_agent,
            evolution_agent=evolution_agent,
            execution_agent=execution_agent,
            context_tracker=context_tracker,
            llm=llm,
            temperature=temperature,
            prompts=prompts
        )
        
        # Initialize feature configuration
        self.feature_config = FeatureConfig(config_path)
        
        # Initialize event bus for component communication
        self.event_bus = EventBus()
        self._register_event_handlers()
        
        # Initialize enhanced components
        self._init_enhanced_components()
        
        logger.info("IntegratedMetaControllerAgent initialized with all enhanced components")
    
    def _init_enhanced_components(self):
        """Initialize all enhanced components based on feature configuration"""
        # Initialize context components
        if self.feature_config.is_enabled("context_consolidation"):
            self.context_consolidation = ContextConsolidationLayer(
                context_tracker=self.context_tracker,
                llm=self.llm
            )
        else:
            self.context_consolidation = None
            
        if self.feature_config.is_enabled("proactive_evolution"):
            self.concept_evolution_detector = ProactiveConceptEvolutionDetector(
                llm=self.llm
            )
        else:
            self.concept_evolution_detector = None
            
        if self.feature_config.is_enabled("concept_recall"):
            self.concept_recall = ConceptRecallSystem(
                context_tracker=self.context_tracker,
                llm=self.llm
            )
        else:
            self.concept_recall = None
        
        # Initialize validation components
        if self.feature_config.is_enabled("logical_coherence"):
            self.tree_of_thought = TreeOfThoughtValidator(
                llm=self.llm,
                max_paths=3,
                max_depth=2
            )
            
            self.coherence_validator = LogicalCoherenceValidator(
                llm=self.llm,
                tree_of_thought_handler=self.tree_of_thought
            )
            
            # Wrap with lazy validator for performance optimization
            self.lazy_coherence_validator = LazyValidator(
                validator_func=self.coherence_validator.validate_coherence,
                threshold=0.85,
                cache_size=100
            )
        else:
            self.tree_of_thought = None
            self.coherence_validator = None
            self.lazy_coherence_validator = None
        
        # Initialize execution components
        if self.feature_config.is_enabled("feedback_execution"):
            self.feedback_modifier = FeedbackBasedExecutionModifier(
                user_profile_manager=getattr(self, 'user_profile_manager', None)
            )
        else:
            self.feedback_modifier = None
            
        if self.feature_config.is_enabled("graph_of_thought"):
            self.graph_analyzer = GraphOfThoughtCodeAnalyzer(
                llm=self.llm,
                dependency_graph=getattr(self.execution_agent, 'dependency_graph', None)
            )
        else:
            self.graph_analyzer = None
            
        if self.feature_config.is_enabled("pal_generation"):
            # Assuming execution_agent has code_generator and code_analyzer attributes
            code_analyzer = getattr(self.execution_agent, 'code_analyzer', None)
            self.pal_generator = PALEnhancedCodeGenerator(
                llm=self.llm,
                code_analyzer=code_analyzer
            )
            # Inject PAL generator into execution agent
            if hasattr(self.execution_agent, 'code_generator'):
                self.execution_agent.code_generator = self.pal_generator
        else:
            self.pal_generator = None
        
        # Initialize optimization components
        if self.feature_config.is_enabled("opro_optimization"):
            self.prompt_optimizer = OPROPromptOptimizer(
                llm=self.llm
            )
            # Register core prompts for optimization
            for prompt_id, prompt_text in self.prompts.items():
                self.prompt_optimizer.register_prompt(
                    prompt_id=prompt_id,
                    initial_prompt=prompt_text,
                    description=f"Prompt for {prompt_id}"
                )
        else:
            self.prompt_optimizer = None
            
        # Always initialize quality monitor
        self.quality_monitor = QualityMonitor(
            alert_threshold=0.2,
            window_size=10
        )
    
    def _register_event_handlers(self):
        """Register event handlers for component communication"""
        # Register concept evolution events
        self.event_bus.subscribe(
            "concept_evolution_detected",
            self._handle_concept_evolution
        )
        
        # Register concept recall events
        self.event_bus.subscribe(
            "critical_concepts_recalled",
            self._handle_concept_recall
        )
        
        # Register validation events
        self.event_bus.subscribe(
            "validation_failed",
            self._handle_validation_failure
        )
        
        # Register user feedback events
        self.event_bus.subscribe(
            "user_feedback_received",
            self._handle_user_feedback
        )
        
        # Register quality alert events
        self.event_bus.subscribe(
            "quality_alert",
            self._handle_quality_alert
        )
    
    def process_query(self, query: str, project_vision: Any = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query with enhanced context awareness and logical validation
        
        Args:
            query: User's input query
            project_vision: Optional project vision for context
            user_id: Optional user ID for personalization
            
        Returns:
            Dictionary with processed response and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {query[:50]}...")
        
        # Record quality metrics from previous processing
        self._record_quality_metrics()
        
        # Step 1: Classify query complexity
        classification_result, success = self.safeguards.execute_with_safeguards(
            "classifier",
            self.query_classifier.classify,
            query
        )
        
        processing_tier = classification_result.get("processing_tier", "standard")
        agent_requirements = classification_result.get("agent_requirements", {})
        
        logger.info(f"Query classified as '{processing_tier}' tier with requirements: {agent_requirements}")
        
        # Step 2: Process the query based on classification
        if processing_tier == "comprehensive":
            # For complex queries, use DecomP with Graph-of-Thought
            if self.feature_config.is_enabled("graph_of_thought") and self._should_use_got(query, classification_result):
                result = self._process_with_got(query, classification_result, project_vision, user_id)
            else:
                result = self._process_with_decomp(query, classification_result, project_vision, user_id)
                
        elif processing_tier == "standard":
            # For standard queries, use Tree-of-Thought if beneficial
            if self.feature_config.is_enabled("tree_of_thought") and self._should_use_tot(query, classification_result):
                result = self._process_with_tot(query, classification_result, project_vision, user_id)
            else:
                result = self._process_standard_query(query, classification_result, project_vision, user_id)
                
        else:
            # For lightweight queries, use the basic approach
            result = self._process_lightweight_query(query, classification_result, user_id)
        
        # Apply final post-processing steps
        result = self._apply_post_processing(result, query, user_id)
        
        # Calculate and record processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        logger.info(f"Query processing completed in {processing_time:.2f} seconds")
        return result
    
    def _process_with_decomp(self, query, classification_result, project_vision=None, user_id=None):
        """Process query using Decomposed Prompting"""
        # Step 1: Break down query into sub-tasks with dependencies
        decomposition = self._decompose_query(query, classification_result)
        
        # Step 2: Create execution plan based on dependencies
        execution_plan = self._create_execution_plan(decomposition)
        
        # Step 3: Execute sub-tasks according to plan
        subtask_results = {}
        for task_id in execution_plan["execution_order"]:
            task = execution_plan["tasks"][task_id]
            
            # Get results of dependencies
            dependency_results = {
                dep_id: subtask_results[dep_id]
                for dep_id in task["dependencies"]
                if dep_id in subtask_results
            }
            
            # Execute sub-task
            subtask_results[task_id] = self._execute_subtask(
                task_id=task_id,
                task=task,
                query=query,
                dependency_results=dependency_results,
                project_vision=project_vision,
                user_id=user_id
            )
        
        # Step 4: Integrate sub-task results into final response
        integrated_result = self._integrate_subtask_results(
            query=query,
            execution_plan=execution_plan,
            subtask_results=subtask_results,
            project_vision=project_vision,
            user_id=user_id
        )
        
        return integrated_result
    
    def _process_with_tot(self, query, classification_result, project_vision=None, user_id=None):
        """Process query using Tree-of-Thought reasoning"""
        # Step 1: Generate multiple reasoning paths
        reasoning_paths = self._generate_reasoning_paths(query, classification_result)
        
        # Step 2: Evaluate each path
        evaluated_paths = []
        for path in reasoning_paths:
            # Process query along this reasoning path
            path_result = self._process_standard_query(
                query=query,
                classification_result=classification_result,
                project_vision=project_vision,
                user_id=user_id,
                reasoning_path=path
            )
            
            # Add original reasoning path to the result
            path_result["reasoning_path"] = path
            
            # Evaluate the quality of this result
            evaluation = self._evaluate_reasoning_path(
                query=query,
                path=path,
                result=path_result
            )
            
            path_result["evaluation"] = evaluation
            evaluated_paths.append(path_result)
        
        # Step 3: Select the best path
        best_path = max(evaluated_paths, key=lambda p: p["evaluation"]["score"])
        
        # Return the result from the best path
        return best_path
    
    def _process_with_got(self, query, classification_result, project_vision=None, user_id=None):
        """Process query using Graph-of-Thought reasoning"""
        # For code-related queries using Graph-of-Thought
        if self.graph_analyzer and "requires_code_generation" in classification_result:
            # Step 1: Get relevant code blocks from context
            code_blocks = self._extract_code_blocks_from_context(query)
            
            # Step 2: Get Graph-of-Thought analysis
            got_analysis = self.graph_analyzer.analyze_with_got(query, code_blocks)
            
            # Step 3: Process with standard approach, but include GOT analysis
            result = self._process_standard_query(
                query=query,
                classification_result=classification_result,
                project_vision=project_vision,
                user_id=user_id,
                got_analysis=got_analysis
            )
            
            # Add GOT analysis to the result
            result["got_analysis"] = got_analysis
            
            return result
        else:
            # Fallback to standard processing if GOT not applicable
            return self._process_standard_query(
                query=query,
                classification_result=classification_result,
                project_vision=project_vision,
                user_id=user_id
            )
    
    def _process_standard_query(
        self,
        query, 
        classification_result, 
        project_vision=None, 
        user_id=None,
        reasoning_path=None,
        got_analysis=None
    ):
        """Process standard tier query with enhanced context awareness"""
        # Step 1: Determine which agents to involve based on necessity scores
        memory_threshold = 0.3
        context_threshold = 0.4
        evolution_threshold = 0.5
        execution_threshold = 0.4
        
        agent_requirements = classification_result.get("agent_requirements", {})
        
        # Step 2: Retrieve relevant context from Memory Agent
        memory_context = []
        if agent_requirements.get("memory", 1.0) >= memory_threshold:
            logger.info("Getting context from Memory Agent")
            memory_context, success = self.safeguards.execute_with_safeguards(
                "memory_agent",
                self._get_memory_context,
                query
            )
        else:
            logger.info("Skipping Memory Agent (below threshold)")
        
        # Step 3: Apply context consolidation if enabled
        if self.context_consolidation and memory_context:
            important_concepts = self.context_tracker.get_high_importance_concepts(threshold=0.5)
            all_concepts = {c["concept"]: c for c in important_concepts}
            
            # Consolidate context to ensure critical concepts are included
            consolidated_context = self.context_consolidation.consolidate_context(
                query=query,
                recent_context=memory_context,
                all_concepts=all_concepts
            )
            
            # Use consolidated context if it adds value
            if len(consolidated_context) > len(memory_context):
                logger.info(f"Context consolidated: added {len(consolidated_context) - len(memory_context)} entries")
                memory_context = consolidated_context
        
        # Step 4: Get context enforcement report
        context_report = {}
        if agent_requirements.get("context", 1.0) >= context_threshold and memory_context:
            logger.info("Getting context enforcement report")
            context_report, success = self.safeguards.execute_with_safeguards(
                "context_agent",
                self._get_context_report,
                query, memory_context
            )
        else:
            logger.info("Skipping Context Enforcement Agent (below threshold)")
        
        # Step 5: Get evolution report
        evolution_report = {}
        if agent_requirements.get("evolution", 1.0) >= evolution_threshold and memory_context:
            logger.info("Getting evolution report")
            evolution_report, success = self.safeguards.execute_with_safeguards(
                "evolution_agent",
                self._get_evolution_report,
                query, memory_context
            )
        else:
            logger.info("Skipping Evolution Agent (below threshold)")
        
        # Step 6: Get execution parameters from user feedback if available
        style_preferences = {}
        structure_preferences = {}
        if self.feedback_modifier and user_id:
            code_request = {
                "language": self._detect_language(query),
                "purpose": query
            }
            style_preferences, structure_preferences = self.feedback_modifier.get_execution_parameters(
                user_id=user_id,
                code_request=code_request
            )
        
        # Step 7: Generate execution output
        execution_output = {}
        if agent_requirements.get("execution", 1.0) >= execution_threshold:
            logger.info("Getting execution output")
            # If using PAL and it's a code generation request
            if self.pal_generator and classification_result.get("requires_code_generation", False):
                execution_output, success = self.safeguards.execute_with_safeguards(
                    "execution_agent",
                    self._get_enhanced_execution_output,
                    query, memory_context, context_report, evolution_report, project_vision,
                    style_preferences, structure_preferences, got_analysis
                )
            else:
                execution_output, success = self.safeguards.execute_with_safeguards(
                    "execution_agent",
                    self._get_execution_output,
                    query, memory_context, context_report, evolution_report, project_vision
                )
        else:
            logger.info("Skipping Execution Agent (below threshold)")
        
        # Step 8: Integrate all outputs into a unified response
        logger.info("Integrating agent outputs")
        integrated_response = self._integrate_outputs(
            query, 
            memory_context, 
            context_report, 
            evolution_report, 
            execution_output,
            reasoning_path
        )
        
        # Step 9: Apply concept evolution detection if enabled
        if self.concept_evolution_detector and memory_context:
            important_concepts = self.context_tracker.get_high_importance_concepts(threshold=0.5)
            all_concepts = {c["concept"]: c for c in important_concepts}
            
            # Detect concept evolution in the generated response
            evolved_concepts = self.concept_evolution_detector.detect_evolution(
                response=integrated_response,
                existing_concepts=all_concepts
            )
            
            # Notify about detected evolution
            if evolved_concepts:
                self.event_bus.publish(
                    event_type="concept_evolution_detected",
                    data={
                        "evolved_concepts": evolved_concepts,
                        "query": query,
                        "response": integrated_response
                    }
                )
        
        # Step 10: Perform coherence validation if enabled
        validation_result = None
        if self.coherence_validator and self.lazy_coherence_validator:
            # Generate cache key based on query and response
            validation_key = self._generate_validation_key(query, integrated_response)
            
            # Validate coherence with caching
            validation_result = self.lazy_coherence_validator.validate(
                key=validation_key,
                query=query,
                response=integrated_response,
                conversation_history=memory_context,
                threshold=0.7
            )
            
            # If validation fails, notify and consider refinement
            if not validation_result.get("is_valid", True):
                self.event_bus.publish(
                    event_type="validation_failed",
                    data={
                        "validation_result": validation_result,
                        "query": query,
                        "response": integrated_response
                    }
                )
                
                # For simplicity, we'll proceed with original response
                # In a full implementation, you'd add refinement logic here
        
        # Step 11: Apply concept recall if enabled
        recall_result = None
        if self.concept_recall and memory_context:
            # Update timestamps for mentioned concepts
            self.concept_recall.update_timestamps(integrated_response)
            
            # Check for critical concepts that should be recalled
            recall_result = self.concept_recall.recall_critical_concepts(
                query=query,
                recent_context=memory_context,
                response=integrated_response
            )
            
            # If critical concepts need to be recalled, notify
            if recall_result:
                self.event_bus.publish(
                    event_type="critical_concepts_recalled",
                    data={
                        "recall_result": recall_result,
                        "query": query,
                        "response": integrated_response
                    }
                )
                
                # For simplicity, we'll proceed with original response
                # In a full implementation, you'd add concept re-injection here
        
        # Step 12: Create conversation entry and store in context tracker
        entry = self._create_conversation_entry(query, integrated_response, memory_context)
        
        # Return the final processed result
        return {
            "query": query,
            "response": integrated_response,
            "processing_tier": "standard",
            "validation": validation_result,
            "recall": recall_result,
            "context": {
                "memory_context_summary": self._summarize_memory_context(memory_context),
                "context_report_summary": self._summarize_context_report(context_report),
                "evolution_report_summary": self._summarize_evolution_report(evolution_report)
            },
            "entry_id": entry.id
        }
    
    def _process_lightweight_query(self, query, classification_result, user_id=None):
        """Process lightweight query with minimal context"""
        # Enhanced lightweight processing with minimal context
        logger.info("Using lightweight processing path")
        
        # Get minimal context if needed
        basic_context = ""
        if classification_result.get("requires_context_history", False):
            # Get only the most recent context entry
            recent_entries = self.context_tracker.find_sequential_entries(
                query=query,
                top_k=1,
                include_sequential_context=False
            )
            
            if recent_entries:
                basic_context = f"Previous message - User: {recent_entries[0].user_input}"
                if recent_entries[0].system_response:
                    basic_context += f"\nSystem: {recent_entries[0].system_response}"
        
        # Generate response using lightweight chain
        # If OPRO optimization is enabled, get optimized prompt
        if self.prompt_optimizer:
            lightweight_prompt = self.prompt_optimizer.get_current_prompt("lightweight_processing")
            if not lightweight_prompt:
                lightweight_prompt = self.lightweight_prompt.template
                
            # Create temporary prompt template
            from langchain.prompts import PromptTemplate
            temp_prompt = PromptTemplate(
                template=lightweight_prompt,
                input_variables=["query", "basic_context"]
            )
            
            # Create temporary chain
            from langchain.chains import LLMChain
            temp_chain = LLMChain(
                llm=self.llm,
                prompt=temp_prompt,
                verbose=True
            )
            
            response = temp_chain.run(
                query=query,
                basic_context=basic_context
            )
        else:
            # Use standard lightweight chain
            response = self.lightweight_chain.run(
                query=query,
                basic_context=basic_context
            )
        
        # Apply concept recall if enabled
        recall_result = None
        if self.concept_recall:
            # Update timestamps for mentioned concepts
            self.concept_recall.update_timestamps(response)
            
            # Check for critical concepts that should be recalled
            recall_result = self.concept_recall.recall_critical_concepts(
                query=query,
                recent_context=[],  # No context for lightweight
                response=response
            )
            
            # If critical concepts need to be recalled, notify
            if recall_result:
                self.event_bus.publish(
                    event_type="critical_concepts_recalled",
                    data={
                        "recall_result": recall_result,
                        "query": query,
                        "response": response
                    }
                )
        
        # Create a minimal conversation entry
        entry = self._create_conversation_entry(query, response, [])
        
        return {
            "query": query,
            "response": response,
            "processing_tier": "lightweight",
            "validation": {
                "is_valid": True,
                "reflection": "Lightweight processing, no detailed validation performed"
            },
            "recall": recall_result,
            "context": {
                "classification": classification_result
            },
            "entry_id": entry.id
        }
    
    def _get_enhanced_execution_output(
        self, 
        query: str, 
        memory_context: List[Any],
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Any = None,
        style_preferences: Dict[str, Any] = None,
        structure_preferences: Dict[str, Any] = None,
        got_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced execution output with PAL and user preferences"""
        # Determine the language of code to generate
        language = self._detect_language(query)
        
        # Extract code requirements from query and context
        requirements = self._extract_requirements(
            query, context_report, evolution_report, project_vision
        )
        
        # Get existing code from context if modifying
        existing_code = None
        if "modify" in query.lower() or "update" in query.lower() or "change" in query.lower():
            code_blocks = self._extract_code_blocks_from_context(query)
            if code_blocks:
                existing_code = code_blocks[0].content
        
        # Generate code using PAL Generator
        generation_result = self.pal_generator.generate_code(
            language=language,
            purpose=query,
            dependencies=None,  # Would extract from context
            existing_code=existing_code,
            requirements=requirements,
            style_preferences=style_preferences,
            structure_preferences=structure_preferences
        )
        
        # Format as execution output
        execution_output = {
            "request_type": "generate" if not existing_code else "modify",
            "code_blocks": [{
                "id": f"block_{int(time.time())}",
                "language": language,
                "content": generation_result["code"],
                "purpose": query
            }],
            "explanation": generation_result.get("explanation", self._generate_code_explanation(generation_result)),
            "execution_result": generation_result.get("execution_result"),
            "iterations": generation_result.get("iterations", 1)
        }
        
        # Add Graph-of-Thought analysis if available
        if got_analysis:
            execution_output["got_analysis"] = got_analysis
        
        return execution_output
    
    def _extract_code_blocks_from_context(self, query: str) -> List[Any]:
        """Extract relevant code blocks from context based on query"""
        # This would be implemented to pull code blocks from context storage
        # For now, returning an empty list as a placeholder
        return []
    
    def _apply_post_processing(self, result: Dict[str, Any], query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Apply final post-processing to the result"""
        # Add processing metadata
        result["timestamp"] = datetime.now().isoformat()
        result["processed_by"] = "IntegratedMetaControllerAgent"
        
        # Add quality monitoring data if available
        if hasattr(self, 'quality_monitor'):
            quality_report = self.quality_monitor.get_quality_report()
            result["quality"] = {
                "overall_health": quality_report.get("overall_health", 0.5),
                "active_alerts": len(quality_report.get("active_alerts", []))
            }
        
        return result
    
    def _generate_validation_key(self, query: str, response: str) -> str:
        """Generate a unique key for validation caching"""
        import hashlib
        combined = f"{query}|{response}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _record_quality_metrics(self):
        """Record quality metrics for monitoring"""
        if not hasattr(self, 'quality_monitor'):
            return
            
        # Record basic metrics
        # In a full implementation, you'd calculate these metrics from actual data
        context_retention = 0.8  # Example value
        logical_coherence = 0.85  # Example value
        execution_success = 0.9  # Example value
        
        self.quality_monitor.record_metric("context_retention", context_retention)
        self.quality_monitor.record_metric("logical_coherence", logical_coherence)
        self.quality_monitor.record_metric("execution_success", execution_success)
    
    def _generate_code_explanation(self, generation_result: Dict[str, Any]) -> str:
        """Generate an explanation for the code"""
        # Simple implementation - in practice, you might use LLM
        code = generation_result.get("code", "")
        language = generation_result.get("language", "")
        
        return f"This {language} code implements the requested functionality."
    
    def _detect_language(self, query: str) -> str:
        """Detect programming language from query"""
        # Simple language detection heuristic
        query_lower = query.lower()
        
        if "python" in query_lower:
            return "python"
        elif "javascript" in query_lower or "js" in query_lower:
            return "javascript"
        elif "typescript" in query_lower or "ts" in query_lower:
            return "typescript"
        elif "java" in query_lower:
            return "java"
        elif "c#" in query_lower or "csharp" in query_lower:
            return "csharp"
        elif "c++" in query_lower or "cpp" in query_lower:
            return "cpp"
        elif "go" in query_lower or "golang" in query_lower:
            return "go"
        elif "ruby" in query_lower:
            return "ruby"
        
        # Default to Python
        return "python"
    
    def _should_use_tot(self, query: str, classification_result: Dict[str, Any]) -> bool:
        """Determine if Tree-of-Thought should be used for this query"""
        # Check complexity indicators
        if classification_result.get("complexity_factors", []):
            complexity_factors = classification_result.get("complexity_factors", [])
            
            # ToT is useful for these types of complexity
            tot_benefiting_factors = [
                "multiple approaches possible",
                "requires logical reasoning",
                "ambiguous requirements",
                "trade-off analysis",
                "requires creativity"
            ]
            
            # Check if any ToT-benefiting factors are present
            for factor in tot_benefiting_factors:
                for complexity in complexity_factors:
                    if factor.lower() in complexity.lower():
                        return True
        
        # Check query for indicators of complex reasoning
        query_lower = query.lower()
        tot_indicators = [
            "compare", "contrast", "evaluate", "analyze", "prioritize",
            "best approach", "options", "alternatives", "pros and cons"
        ]
        
        if any(indicator in query_lower for indicator in tot_indicators):
            return True
        
        return False
    
    def _should_use_got(self, query: str, classification_result: Dict[str, Any]) -> bool:
        """Determine if Graph-of-Thought should be used for this query"""
        # Check if this is a code-related query
        if classification_result.get("requires_code_generation", False):
            return True
            
        # Check query for code analysis indicators
        query_lower = query.lower()
        got_indicators = [
            "analyze code", "code structure", "refactor", "architecture",
            "dependencies", "coupling", "cohesion", "code quality"
        ]
        
        if any(indicator in query_lower for indicator in got_indicators):
            return True
            
        return False
    
    def process_feedback(self, feedback: str, user_id: Optional[str] = None, entry_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user feedback using Skeleton-of-Thought reasoning
        
        Args:
            feedback: User feedback text
            user_id: Optional user ID for personalization
            entry_id: Optional ID of entry feedback applies to
            
        Returns:
            Feedback processing result
        """
        # Publish feedback event
        self.event_bus.publish(
            event_type="user_feedback_received",
            data={
                "feedback": feedback,
                "user_id": user_id,
                "entry_id": entry_id
            }
        )
        
        # Process with Skeleton-of-Thought reasoning
        feedback_result = self.process_feedback_with_skeleton(
            feedback_text=feedback,
            user_id=user_id,
            entry_id=entry_id
        )
        
        return feedback_result
    
    # Event handlers
    def _handle_concept_evolution(self, data: Dict[str, Any]) -> None:
        """Handle concept evolution event"""
        evolved_concepts = data.get("evolved_concepts", {})
        
        # In a full implementation, you would update the concept registry
        # with the evolved concepts
        logger.info(f"Handling concept evolution for {len(evolved_concepts)} concepts")
    
    def _handle_concept_recall(self, data: Dict[str, Any]) -> None:
        """Handle concept recall event"""
        recall_result = data.get("recall_result", {})
        
        # In a full implementation, you would modify the response
        # to reincorporate the recalled concepts
        logger.info(f"Handling concept recall for {len(recall_result.get('critical_concepts', []))} concepts")
    
    def _handle_validation_failure(self, data: Dict[str, Any]) -> None:
        """Handle validation failure event"""
        validation_result = data.get("validation_result", {})
        
        # In a full implementation, you would refine the response
        # to address the validation issues
        logger.info(f"Handling validation failure with {len(validation_result.get('issues', []))} issues")
    
    def _handle_user_feedback(self, data: Dict[str, Any]) -> None:
        """Handle user feedback event"""
        feedback = data.get("feedback", "")
        user_id = data.get("user_id")
        
        # Update feedback-based execution parameters if enabled
        if self.feedback_modifier and user_id:
            # This would extract and store user preferences from feedback
            # For now, just log the action
            logger.info(f"Updating user preferences from feedback for user {user_id}")
    
    def _handle_quality_alert(self, data: Dict[str, Any]) -> None:
        """Handle quality alert event"""
        alert = data.get("alert", {})
        
        # In a full implementation, you would take corrective action
        # based on the quality alert
        logger.info(f"Handling quality alert: {alert.get('type', 'unknown')}")

# Factory function to create an integrated agent
def create_integrated_agent(
    memory_agent: Any,
    context_agent: Any,
    evolution_agent: Any,
    execution_agent: Any,
    context_tracker: Any,
    llm: Any,
    config_path: Optional[str] = None
) -> IntegratedMetaControllerAgent:
    """Create a fully initialized integrated meta-controller agent"""
    return IntegratedMetaControllerAgent(
        memory_agent=memory_agent,
        context_agent=context_agent,
        evolution_agent=evolution_agent,
        execution_agent=execution_agent,
        context_tracker=context_tracker,
        llm=llm,
        config_path=config_path
    )
