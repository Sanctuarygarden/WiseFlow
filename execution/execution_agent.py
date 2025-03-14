import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
import re
import hashlib
from collections import defaultdict

# Import LangChain components
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Local imports
from sequential_context_tracking import ConversationEntry, PriorityTier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionAgent")

# ============= Code Block & Dependency Structures =============

@dataclass
class CodeBlock:
    """
    Enhanced representation of a code block with dependency tracking
    
    This structure:
    1. Tracks content, language, and metadata
    2. Maintains a unique ID and hash for change detection
    3. Tracks dependencies to other code blocks
    4. Records purpose and functional intent
    """
    id: str  # Unique identifier for this code block
    content: str  # The actual code
    language: str  # Programming language
    file_path: Optional[str] = None  # Logical file path if part of a project
    hash: str = field(default="")  # Content hash for change detection
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Dependency tracking
    dependencies: List[str] = field(default_factory=list)  # IDs of blocks this depends on
    dependents: List[str] = field(default_factory=list)  # IDs of blocks that depend on this
    
    # Functional description
    purpose: str = ""  # What this code does
    api_surface: Dict[str, Any] = field(default_factory=dict)  # Functions/classes exposed
    
    # Revision tracking
    revision: int = 1  # Revision number
    parent_id: Optional[str] = None  # ID of previous version if this is a revision
    
    # Extracted components
    functions: List[Dict[str, Any]] = field(default_factory=list)  # Functions defined in this block
    classes: List[Dict[str, Any]] = field(default_factory=list)  # Classes defined in this block
    imports: List[Dict[str, Any]] = field(default_factory=list)  # Imports used in this block
    variables: List[Dict[str, Any]] = field(default_factory=list)  # Global variables defined
    
    def __post_init__(self):
        """Initialize derived fields if not provided"""
        if not self.hash and self.content:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeBlock':
        """Create instance from dictionary"""
        return cls(**data)
    
    def update_content(self, new_content: str) -> 'CodeBlock':
        """
        Create a new version of this code block with updated content
        
        Returns:
            New CodeBlock instance with incremented revision
        """
        new_hash = hashlib.md5(new_content.encode()).hexdigest()
        
        # If content hasn't changed, don't create a new revision
        if new_hash == self.hash:
            return self
        
        # Create new block as next revision
        new_id = f"{self.id}_r{self.revision + 1}"
        
        return CodeBlock(
            id=new_id,
            content=new_content,
            language=self.language,
            file_path=self.file_path,
            hash=new_hash,
            dependencies=self.dependencies.copy(),
            dependents=self.dependents.copy(),
            purpose=self.purpose,
            api_surface=self.api_surface.copy(),
            revision=self.revision + 1,
            parent_id=self.id,
            functions=self.functions.copy(),
            classes=self.classes.copy(),
            imports=self.imports.copy(),
            variables=self.variables.copy()
        )
    
    def is_compatible_with(self, other_block: 'CodeBlock') -> bool:
        """Check if this block is API-compatible with another block"""
        # If languages don't match, they're not compatible
        if self.language != other_block.language:
            return False
            
        # If no API surface defined, we can't tell
        if not self.api_surface or not other_block.api_surface:
            return True
            
        # Check if all functions in other block exist in this one with compatible signatures
        for func_name, func_sig in other_block.api_surface.get("functions", {}).items():
            if func_name not in self.api_surface.get("functions", {}):
                return False
                
            # Simple signature compatibility check (just param count for now)
            other_params = func_sig.get("params", [])
            this_params = self.api_surface.get("functions", {}).get(func_name, {}).get("params", [])
            
            # If other block has more required params, they're not compatible
            required_other = sum(1 for p in other_params if not p.get("optional", False))
            required_this = sum(1 for p in this_params if not p.get("optional", False))
            
            if required_other > required_this:
                return False
        
        return True

@dataclass
class CodeDependencyGraph:
    """
    Maintains relationships between code blocks for dependency tracking
    
    This component:
    1. Tracks dependencies between code blocks
    2. Identifies affected blocks when changes occur
    3. Validates changes for dependency compatibility
    4. Provides visualization of the dependency structure
    """
    blocks: Dict[str, CodeBlock] = field(default_factory=dict)
    current_blocks: Dict[str, str] = field(default_factory=dict)  # file_path -> current block_id
    
    def add_block(self, block: CodeBlock) -> None:
        """Add a code block to the graph"""
        self.blocks[block.id] = block
        
        # If this has a file path, update current pointer
        if block.file_path:
            self.current_blocks[block.file_path] = block.id
    
    def get_current_block(self, file_path: str) -> Optional[CodeBlock]:
        """Get the current version of a code block by file path"""
        if file_path in self.current_blocks:
            block_id = self.current_blocks[file_path]
            return self.blocks.get(block_id)
        return None
    
    def get_block(self, block_id: str) -> Optional[CodeBlock]:
        """Get a code block by ID"""
        return self.blocks.get(block_id)
    
    def get_blocks_by_language(self, language: str) -> List[CodeBlock]:
        """Get all current blocks of a specific language"""
        result = []
        for file_path, block_id in self.current_blocks.items():
            block = self.blocks.get(block_id)
            if block and block.language == language:
                result.append(block)
        return result
    
    def add_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """
        Add a dependency relationship between blocks
        
        Args:
            dependent_id: ID of the block that depends on the other
            dependency_id: ID of the block being depended upon
            
        Returns:
            Success flag
        """
        if dependent_id not in self.blocks or dependency_id not in self.blocks:
            return False
            
        # Add to dependent's dependencies
        dependent = self.blocks[dependent_id]
        if dependency_id not in dependent.dependencies:
            dependent.dependencies.append(dependency_id)
            
        # Add to dependency's dependents
        dependency = self.blocks[dependency_id]
        if dependent_id not in dependency.dependents:
            dependency.dependents.append(dependent_id)
            
        return True
    
    def get_dependencies(self, block_id: str, recursive: bool = False) -> List[str]:
        """
        Get dependencies of a block
        
        Args:
            block_id: ID of the block to get dependencies for
            recursive: Whether to include indirect dependencies
            
        Returns:
            List of dependency block IDs
        """
        if block_id not in self.blocks:
            return []
            
        block = self.blocks[block_id]
        
        if not recursive:
            return block.dependencies
            
        # Recursive dependencies (using depth-first search)
        visited = set()
        result = []
        
        def visit(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            current_block = self.blocks.get(current_id)
            if not current_block:
                return
                
            for dep_id in current_block.dependencies:
                if dep_id not in visited:
                    result.append(dep_id)
                    visit(dep_id)
        
        visit(block_id)
        return result
    
    def get_dependents(self, block_id: str, recursive: bool = False) -> List[str]:
        """
        Get blocks that depend on this one
        
        Args:
            block_id: ID of the block to get dependents for
            recursive: Whether to include indirect dependents
            
        Returns:
            List of dependent block IDs
        """
        if block_id not in self.blocks:
            return []
            
        block = self.blocks[block_id]
        
        if not recursive:
            return block.dependents
            
        # Recursive dependents (using depth-first search)
        visited = set()
        result = []
        
        def visit(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            current_block = self.blocks.get(current_id)
            if not current_block:
                return
                
            for dep_id in current_block.dependents:
                if dep_id not in visited:
                    result.append(dep_id)
                    visit(dep_id)
        
        visit(block_id)
        return result
    
    def check_update_impact(self, block_id: str, new_content: str) -> Dict[str, Any]:
        """
        Check the impact of updating a code block
        
        Args:
            block_id: ID of the block to update
            new_content: New content for the block
            
        Returns:
            Dictionary with impact analysis
        """
        if block_id not in self.blocks:
            return {
                "valid": False,
                "reason": "Block not found",
                "affected_blocks": []
            }
            
        current_block = self.blocks[block_id]
        
        # Create temporary new block to analyze
        new_hash = hashlib.md5(new_content.encode()).hexdigest()
        
        # If content hasn't changed, no impact
        if new_hash == current_block.hash:
            return {
                "valid": True,
                "reason": "No changes detected",
                "affected_blocks": []
            }
            
        # Create new block for analysis
        new_block = CodeBlock(
            id=f"{block_id}_analysis",
            content=new_content,
            language=current_block.language,
            file_path=current_block.file_path,
            hash=new_hash
        )
        
        # Get dependents that might be affected
        dependents = self.get_dependents(block_id)
        affected_blocks = []
        
        # Simple compatibility check (would be more sophisticated in real impl)
        # Just check if block still defines the same functions/classes
        for dependent_id in dependents:
            dependent = self.blocks.get(dependent_id)
            if dependent:
                affected_blocks.append({
                    "id": dependent_id,
                    "file_path": dependent.file_path,
                    "purpose": dependent.purpose,
                    "impact": "potential"  # We'd need more analysis to determine actual impact
                })
        
        return {
            "valid": True,
            "reason": "Update impacts dependents",
            "affected_blocks": affected_blocks
        }
    
    def update_block(self, block_id: str, new_content: str) -> Tuple[CodeBlock, List[str]]:
        """
        Update a code block and track the impact
        
        Args:
            block_id: ID of the block to update
            new_content: New content for the block
            
        Returns:
            Tuple of (new_block, affected_dependent_ids)
        """
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found")
            
        current_block = self.blocks[block_id]
        
        # Create new version
        new_block = current_block.update_content(new_content)
        
        # If no changes, return current block
        if new_block.id == current_block.id:
            return current_block, []
            
        # Add new block to graph
        self.blocks[new_block.id] = new_block
        
        # Update current pointer if this is a file-linked block
        if new_block.file_path:
            self.current_blocks[new_block.file_path] = new_block.id
            
        # Get affected dependents
        affected_dependents = current_block.dependents.copy()
        
        # Transfer dependencies to new block
        for dep_id in current_block.dependencies:
            dependency = self.blocks.get(dep_id)
            if dependency:
                # Remove old block from dependency's dependents
                if current_block.id in dependency.dependents:
                    dependency.dependents.remove(current_block.id)
                # Add new block to dependency's dependents
                if new_block.id not in dependency.dependents:
                    dependency.dependents.append(new_block.id)
        
        # Transfer dependents to new block
        for dep_id in current_block.dependents:
            dependent = self.blocks.get(dep_id)
            if dependent:
                # Remove old block from dependent's dependencies
                if current_block.id in dependent.dependencies:
                    dependent.dependencies.remove(current_block.id)
                # Add new block to dependent's dependencies
                if new_block.id not in dependent.dependencies:
                    dependent.dependencies.append(new_block.id)
        
        return new_block, affected_dependents
    
    def visualize_dependencies(self, block_id: Optional[str] = None) -> str:
        """
        Generate a Mermaid graph for visualizing dependencies
        
        Args:
            block_id: Optional root block to visualize from
            
        Returns:
            Mermaid diagram string
        """
        mermaid = ["graph TD"]
        
        # Track nodes and edges to avoid duplicates
        nodes = set()
        edges = set()
        
        # Helper to add a node
        def add_node(node_id, label, style=None):
            if node_id in nodes:
                return
                
            nodes.add(node_id)
            node_escaped = node_id.replace("-", "_").replace(".", "_")
            
            node_str = f'    {node_escaped}["{label}"]'
            if style:
                node_str += f" :::{style}"
                
            mermaid.append(node_str)
        
        # Helper to add an edge
        def add_edge(from_id, to_id, label=None):
            if (from_id, to_id) in edges:
                return
                
            edges.add((from_id, to_id))
            from_escaped = from_id.replace("-", "_").replace(".", "_")
            to_escaped = to_id.replace("-", "_").replace(".", "_")
            
            if label:
                mermaid.append(f'    {from_escaped} --> |"{label}"| {to_escaped}')
            else:
                mermaid.append(f'    {from_escaped} --> {to_escaped}')
        
        # If specific block ID provided, only show that with direct dependencies and dependents
        if block_id:
            block = self.blocks.get(block_id)
            if not block:
                return "graph TD\n    A[Block not found]"
                
            # Add the central block
            add_node(block_id, f"{block.file_path or block.id} (v{block.revision})", "focus")
            
            # Add dependencies
            for dep_id in block.dependencies:
                dep = self.blocks.get(dep_id)
                if dep:
                    label = dep.file_path.split("/")[-1] if dep.file_path else dep.id
                    add_node(dep_id, f"{label} (v{dep.revision})", "dependency")
                    add_edge(block_id, dep_id, "depends on")
            
            # Add dependents
            for dep_id in block.dependents:
                dep = self.blocks.get(dep_id)
                if dep:
                    label = dep.file_path.split("/")[-1] if dep.file_path else dep.id
                    add_node(dep_id, f"{label} (v{dep.revision})", "dependent")
                    add_edge(dep_id, block_id, "depends on")
        else:
            # Show all current blocks and their relationships
            for file_path, block_id in self.current_blocks.items():
                block = self.blocks.get(block_id)
                if not block:
                    continue
                    
                # Add the block
                label = file_path.split("/")[-1] if file_path else block_id
                add_node(block_id, f"{label} (v{block.revision})")
                
                # Add dependencies
                for dep_id in block.dependencies:
                    if dep_id in self.blocks:
                        add_edge(block_id, dep_id)
        
        # Add styling
        mermaid.append("    classDef focus fill:#f96,stroke:#333,stroke-width:2px")
        mermaid.append("    classDef dependency fill:#9cf,stroke:#333")
        mermaid.append("    classDef dependent fill:#fc9,stroke:#333")
        
        return "\n".join(mermaid)

# ============= Code Analysis Tools =============

class CodeAnalyzer:
    """
    Analyzes code to extract structure, dependencies, and functional elements
    
    This component:
    1. Extracts functions, classes, and imports
    2. Identifies dependencies between code elements
    3. Analyzes API surface for compatibility checking
    4. Detects potential issues and regressions
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        logger.info("CodeAnalyzer initialized")
    
    def analyze_block(self, content: str, language: str) -> Dict[str, Any]:
        """
        Analyze a code block to extract structure and dependencies
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Dictionary with analysis results
        """
        # If LLM available, use it for deeper analysis
        if self.llm:
            return self._analyze_with_llm(content, language)
            
        # Basic analysis without LLM
        return self._basic_code_analysis(content, language)
    
    def extract_api_surface(self, content: str, language: str) -> Dict[str, Any]:
        """
        Extract the API surface (functions, classes, etc.) from a code block
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Dictionary with API elements
        """
        api_surface = {
            "functions": {},
            "classes": {},
            "variables": {}
        }
        
        # Use language-specific extraction
        if language == "python":
            # Extract function definitions
            function_matches = re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\s*->.*?)?:', content)
            
            for match in function_matches:
                func_name = match.group(1)
                params_str = match.group(2).strip()
                
                # Parse parameters
                params = []
                if params_str:
                    param_parts = params_str.split(',')
                    for part in param_parts:
                        part = part.strip()
                        if part:
                            # Check if parameter has default value (is optional)
                            has_default = '=' in part
                            param_name = part.split('=')[0].strip()
                            # Remove type hints
                            if ':' in param_name:
                                param_name = param_name.split(':')[0].strip()
                                
                            params.append({
                                "name": param_name,
                                "optional": has_default
                            })
                
                api_surface["functions"][func_name] = {
                    "params": params,
                    "has_return": "->" in match.group(0)
                }
            
            # Extract class definitions
            class_matches = re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(.*?\))?:', content)
            
            for match in class_matches:
                class_name = match.group(1)
                api_surface["classes"][class_name] = {
                    "methods": {}  # Would need more sophisticated parsing for methods
                }
        
        elif language == "javascript" or language == "typescript":
            # Extract function definitions (including arrow functions)
            function_matches = re.finditer(r'(?:function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\((.*?)\)|const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:function)?\s*\((.*?)\)|const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\((.*?)\)\s*=>)', content)
            
            for match in function_matches:
                if match.group(1):  # Standard function
                    func_name = match.group(1)
                    params_str = match.group(2)
                elif match.group(3):  # Function expression
                    func_name = match.group(3)
                    params_str = match.group(4)
                else:  # Arrow function
                    func_name = match.group(5)
                    params_str = match.group(6)
                
                # Parse parameters
                params = []
                if params_str:
                    param_parts = params_str.split(',')
                    for part in param_parts:
                        part = part.strip()
                        if part:
                            # Check if parameter has default value (is optional)
                            has_default = '=' in part
                            param_name = part.split('=')[0].strip()
                            # Remove type hints
                            if ':' in param_name:
                                param_name = param_name.split(':')[0].strip()
                                
                            params.append({
                                "name": param_name,
                                "optional": has_default
                            })
                
                api_surface["functions"][func_name] = {
                    "params": params
                }
            
            # Extract class definitions
            class_matches = re.finditer(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$]*))?', content)
            
            for match in class_matches:
                class_name = match.group(1)
                api_surface["classes"][class_name] = {
                    "methods": {},  # Would need more sophisticated parsing for methods
                    "extends": match.group(2) if match.group(2) else None
                }
        
        return api_surface
    
    def extract_dependencies(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract dependencies (imports, requires) from code
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            List of dependency descriptors
        """
        dependencies = []
        
        # Handle different languages
        if language == "python":
            # Extract import statements
            import_matches = re.finditer(r'(?:from\s+([\w.]+)\s+import\s+(?:[^#\n]+)|import\s+([\w.,\s]+))', content)
            
            for match in function_matches:
                if match.group(1):  # from X import Y
                    module = match.group(1)
                    dependencies.append({
                        "type": "import",
                        "module": module,
                        "elements": []  # Would need more parsing for specific elements
                    })
                elif match.group(2):  # import X
                    modules = [m.strip() for m in match.group(2).split(',')]
                    for module in modules:
                        dependencies.append({
                            "type": "import",
                            "module": module,
                            "elements": []
                        })
        
        elif language == "javascript" or language == "typescript":
            # Extract import statements
            import_matches = re.finditer(r'import\s+(?:{[^}]*}|[^{]*)\s+from\s+[\'"]([^\'"]+)[\'"]', content)
            
            for match in import_matches:
                module = match.group(1)
                dependencies.append({
                    "type": "import",
                    "module": module,
                    "elements": []  # Would need more parsing for specific elements
                })
            
            # Extract require statements
            require_matches = re.finditer(r'(?:const|let|var)\s+(?:[^=]+)\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\)', content)
            
            for match in require_matches:
                module = match.group(1)
                dependencies.append({
                    "type": "require",
                    "module": module,
                    "elements": []
                })
        
        return dependencies
    
    def detect_function_calls(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Detect function calls to help identify runtime dependencies
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            List of function call descriptors
        """
        # Simple implementation for demonstration
        function_calls = []
        
        # Use regular expressions to find function calls (limited accuracy)
        if language in ["python", "javascript", "typescript"]:
            # This pattern looks for function calls like functionName(args)
            call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            
            for match in re.finditer(call_pattern, content):
                func_name = match.group(1)
                
                # Skip common keywords that look like function calls
                if func_name in ["if", "for", "while", "switch", "catch"]:
                    continue
                    
                function_calls.append({
                    "function": func_name,
                    "position": match.start()
                })
        
        return function_calls
    
    def _basic_code_analysis(self, content: str, language: str) -> Dict[str, Any]:
        """Basic code analysis without LLM"""
        api_surface = self.extract_api_surface(content, language)
        dependencies = self.extract_dependencies(content, language)
        function_calls = self.detect_function_calls(content, language)
        
        # Extract functions and classes
        functions = []
        for name, data in api_surface["functions"].items():
            functions.append({
                "name": name,
                "params": data["params"],
                "has_return": data.get("has_return", False)
            })
            
        classes = []
        for name, data in api_surface["classes"].items():
            classes.append({
                "name": name,
                "methods": list(data.get("methods", {}).keys()),
                "extends": data.get("extends")
            })
            
        # Extract imports
        imports = []
        for dep in dependencies:
            if dep.get("type") in ["import", "require"]:
                imports.append({
                    "module": dep.get("module", ""),
                    "elements": dep.get("elements", [])
                })
        
        return {
            "api_surface": api_surface,
            "dependencies": dependencies,
            "function_calls": function_calls,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "variables": []  # Would need more analysis for this
        }
    
    def _analyze_with_llm(self, content: str, language: str) -> Dict[str, Any]:
        """Use LLM for more sophisticated code analysis"""
        # Basic analysis as a starting point
        basic_analysis = self._basic_code_analysis(content, language)
        
        # Create analysis prompt
        prompt = f"""
        Analyze this {language} code and extract its structure:
        
        ```{language}
        {content}
        ```
        
        Please extract:
        1. Functions with their parameters and return types
        2. Classes with their methods and inheritance
        3. Imports and external dependencies
        4. Global variables
        5. The main purpose of this code block
        6. API surface (functions/classes exposed for external use)
        
        Return your analysis as a JSON object with the following structure:
        {{
            "functions": [{{ "name": "func_name", "params": ["param1", "param2"], "returns": "description" }}],
            "classes": [{{ "name": "ClassName", "methods": ["method1", "method2"], "extends": "ParentClass" }}],
            "imports": [{{ "module": "module_name", "elements": ["element1", "element2"] }}],
            "variables": [{{ "name": "var_name", "type": "type_if_known" }}],
            "purpose": "Brief description of what this code does",
            "api_surface": "Functions/classes that appear to be exposed for external use"
        }}
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                llm_analysis = json.loads(json_match.group(0))
                
                # Merge with basic analysis (prefer LLM for higher-level understanding)
                merged = {**basic_analysis}
                
                # Update with LLM insights
                if "functions" in llm_analysis and llm_analysis["functions"]:
                    merged["functions"] = llm_analysis["functions"]
                
                if "classes" in llm_analysis and llm_analysis["classes"]:
                    merged["classes"] = llm_analysis["classes"]
                    
                if "variables" in llm_analysis and llm_analysis["variables"]:
                    merged["variables"] = llm_analysis["variables"]
                    
                if "purpose" in llm_analysis:
                    merged["purpose"] = llm_analysis["purpose"]
                    
                return merged
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback to basic analysis
        return basic_analysis

class CodeGenerator:
    """
    Generates and modifies code based on specifications and context
    
    This component:
    1. Generates code from text descriptions
    2. Modifies existing code based on requirements
    3. Ensures compatibility with dependencies
    4. Handles different programming languages
    """
    def __init__(self, llm: Any):
        self.llm = llm
        self.analyzer = CodeAnalyzer(llm)
        logger.info("CodeGenerator initialized")
    
    def generate_code(
        self,
        language: str,
        purpose: str,
        dependencies: List[CodeBlock] = None,
        existing_code: str = None,
        requirements: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code based on specifications
        
        Args:
            language: Programming language to generate
            purpose: Description of what the code should do
            dependencies: Related code blocks to maintain compatibility with
            existing_code: Existing code to modify (if any)
            requirements: Specific requirements the code must meet
            
        Returns:
            Dictionary with generated code and metadata
        """
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
        
        # Create generation prompt
        operation = "modify" if existing_code else "generate"
        prompt = f"""
        You are a skilled programmer. I need you to {operation} {language} code for the following purpose:
        
        PURPOSE:
        {purpose}
        
        {dependencies_text}
        
        {requirements_text}
        
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
    
    def modify_code(
        self,
        original_block: CodeBlock,
        modification_description: str,
        dependencies: List[CodeBlock] = None,
        requirements: List[str] = None
    ) -> Dict[str, Any]:
        """
        Modify existing code based on requirements
        
        Args:
            original_block: Original code block to modify
            modification_description: Description of what to change
            dependencies: Related code blocks to maintain compatibility with
            requirements: Specific requirements the modification must meet
            
        Returns:
            Dictionary with modified code and metadata
        """
        # This is essentially a specialized call to generate_code with existing content
        return self.generate_code(
            language=original_block.language,
            purpose=f"Modify the code to {modification_description}. Original purpose: {original_block.purpose}",
            dependencies=dependencies,
            existing_code=original_block.content,
            requirements=requirements
        )
    
    def implement_interface(
        self,
        interface_description: str,
        language: str,
        dependent_blocks: List[CodeBlock] = None
    ) -> Dict[str, Any]:
        """
        Generate code that implements a specified interface
        
        Args:
            interface_description: Description of the interface to implement
            language: Programming language to use
            dependent_blocks: Blocks that will use this interface
            
        Returns:
            Dictionary with generated code and metadata
        """
        # Format dependents for context
        dependents_text = ""
        if dependent_blocks:
            dependents_text = "MODULES THAT WILL USE THIS INTERFACE:\n"
            for i, dep in enumerate(dependent_blocks):
                dependents_text += f"Module {i+1} ({dep.file_path or 'unnamed'}):\n"
                dependents_text += f"```{dep.language}\n{dep.content}\n```\n\n"
        
        # Create implementation prompt
        prompt = f"""
        You are a skilled programmer. I need you to implement an interface in {language} with the following specification:
        
        INTERFACE SPECIFICATION:
        {interface_description}
        
        {dependents_text}
        
        Please provide:
        1. A clean implementation of the interface in {language}
        2. All necessary methods as specified
        3. Good error handling and comments
        4. Compatibility with the modules that will use this interface
        
        Return just the code itself.
        """
        
        generated_code = self.llm.predict(prompt)
        
        # Clean up response to extract just the code
        code = self._extract_code_from_response(generated_code, language)
        
        # Analyze the generated code
        analysis = self.analyzer.analyze_block(code, language)
        
        return {
            "code": code,
            "language": language,
            "purpose": f"Implements interface: {interface_description.split('\n')[0]}",
            "analysis": analysis
        }
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code block from LLM response"""
        # Look for code blocks with language marker
        language_pattern = f"```{language}(.*?)```"
        language_matches = re.findall(language_pattern, response, re.DOTALL)
        
        if language_matches:
            return language_matches[0].strip()
            
        # Look for generic code blocks
        generic_pattern = r"```(.*?)```"
        generic_matches = re.findall(generic_pattern, response, re.DOTALL)
        
        if generic_matches:
            return generic_matches[0].strip()
            
        # If no code blocks found, return the whole response
        return response.strip()

# ============= Execution Agent =============

class ExecutionAgent:
    """
    Agent responsible for generating and refining code based on project vision
    
    This agent:
    1. Generates code based on requirements and context
    2. Tracks dependencies between code blocks
    3. Ensures modifications maintain compatibility
    4. Provides code analysis and suggestions
    """
    def __init__(self, llm, context_tracker: Any):
        self.llm = llm
        self.context_tracker = context_tracker
        self.dependency_graph = CodeDependencyGraph()
        self.code_analyzer = CodeAnalyzer(llm)
        self.code_generator = CodeGenerator(llm)
        
        # Next block ID counter
        self.next_block_id = 1
        
        logger.info("Enhanced Execution Agent initialized")
    
    def process(
        self,
        query: str,
        context: List[ConversationEntry],
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate or refine code based on context and reports
        
        Args:
            query: User's input query
            context: Conversation context
            context_report: Report from Context Enforcement Agent
            evolution_report: Report from Evolution Agent
            project_vision: Optional project vision
            
        Returns:
            Dictionary with execution results
        """
        # Determine if this is a code generation/modification request
        code_request = self._analyze_code_request(query)
        
        # Get existing code blocks from context
        existing_code = self._extract_existing_code(context)
        
        # Create dependency graph from existing code blocks
        self._build_dependency_graph(existing_code)
        
        # Process based on request type
        if code_request["type"] == "generate":
            result = self._handle_generation_request(
                query, 
                code_request, 
                existing_code, 
                context_report, 
                evolution_report,
                project_vision
            )
        elif code_request["type"] == "modify":
            result = self._handle_modification_request(
                query, 
                code_request, 
                existing_code, 
                context_report, 
                evolution_report,
                project_vision
            )
        elif code_request["type"] == "analyze":
            result = self._handle_analysis_request(
                query, 
                code_request, 
                existing_code, 
                context_report, 
                evolution_report,
                project_vision
            )
        else:
            # Default to generation
            result = self._handle_generation_request(
                query, 
                code_request, 
                existing_code, 
                context_report, 
                evolution_report,
                project_vision
            )
        
        return result
    
    def _analyze_code_request(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine code request type and details"""
        # Simplified implementation - in production would use more sophisticated analysis
        request = {
            "type": "unknown",
            "details": {}
        }
        
        # Check for generation request
        generation_patterns = [
            r"(?i)generate (code|a script|program|function|class) (for|to) (.+)",
            r"(?i)create (code|a script|program|function|class) (for|to) (.+)",
            r"(?i)write (code|a script|program|function|class) (for|to) (.+)",
            r"(?i)implement (code|a script|program|function|class) (for|to) (.+)"
        ]
        
        for pattern in generation_patterns:
            match = re.search(pattern, query)
            if match:
                request["type"] = "generate"
                request["details"]["artifact_type"] = match.group(1)
                request["details"]["purpose"] = match.group(3)
                
                # Attempt to determine language
                request["details"]["language"] = self._detect_language(query)
                break
                
        # Check for modification request
        if request["type"] == "unknown":
            modification_patterns = [
                r"(?i)modify (the|this) (code|script|program|function|class) to (.+)",
                r"(?i)update (the|this) (code|script|program|function|class) to (.+)",
                r"(?i)change (the|this) (code|script|program|function|class) to (.+)",
                r"(?i)refactor (the|this) (code|script|program|function|class) to (.+)"
            ]
            
            for pattern in modification_patterns:
                match = re.search(pattern, query)
                if match:
                    request["type"] = "modify"
                    request["details"]["artifact_type"] = match.group(2)
                    request["details"]["modification"] = match.group(3)
                    
                    # Attempt to determine language
                    request["details"]["language"] = self._detect_language(query)
                    break
        
        # Check for analysis request
        if request["type"] == "unknown":
            analysis_patterns = [
                r"(?i)analyze (the|this) (code|script|program|function|class)",
                r"(?i)review (the|this) (code|script|program|function|class)",
                r"(?i)explain (the|this) (code|script|program|function|class)",
                r"(?i)check (the|this) (code|script|program|function|class)"
            ]
            
            for pattern in analysis_patterns:
                match = re.search(pattern, query)
                if match:
                    request["type"] = "analyze"
                    request["details"]["artifact_type"] = match.group(2)
                    
                    # Attempt to determine language
                    request["details"]["language"] = self._detect_language(query)
                    break
        
        # Default to generate if still unknown
        if request["type"] == "unknown":
            request["type"] = "generate"
            request["details"]["inferred"] = True
            request["details"]["language"] = self._detect_language(query)
            
        return request
    
    def _detect_language(self, query: str) -> str:
        """Detect programming language from query"""
        # Check for explicit language mentions
        language_patterns = {
            "python": [r"(?i)\bpython\b"],
            "javascript": [r"(?i)\bjavascript\b", r"(?i)\bjs\b"],
            "typescript": [r"(?i)\btypescript\b", r"(?i)\bts\b"],
            "java": [r"(?i)\bjava\b"],
            "c#": [r"(?i)\bc#\b", r"(?i)\bcsharp\b"],
            "c++": [r"(?i)\bc\+\+\b", r"(?i)\bcpp\b"],
            "go": [r"(?i)\bgo\b", r"(?i)\bgolang\b"],
            "ruby": [r"(?i)\bruby\b"],
            "php": [r"(?i)\bphp\b"],
            "swift": [r"(?i)\bswift\b"],
            "kotlin": [r"(?i)\bkotlin\b"],
            "rust": [r"(?i)\brust\b"],
            "scala": [r"(?i)\bscala\b"],
            "r": [r"(?i)\br( |-)language\b", r"(?i)\br( |-)script\b"]
        }
        
        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return language
        
        # Default to Python if no language specified
        return "python"
    
    def _extract_existing_code(self, context: List[ConversationEntry]) -> List[Dict[str, Any]]:
        """Extract code blocks from context"""
        code_blocks = []
        
        for entry in context:
            code_blocks.extend(entry.code_blocks)
            
        return code_blocks
    
    def _build_dependency_graph(self, code_blocks: List[Dict[str, Any]]) -> None:
        """Build or update dependency graph from code blocks"""
        # Convert basic code blocks to CodeBlock objects and add to graph
        for block_data in code_blocks:
            # Skip if already in graph
            if block_data.get("id") in self.dependency_graph.blocks:
                continue
                
            # Create CodeBlock object
            block = CodeBlock(
                id=block_data.get("id", f"block_{self.next_block_id}"),
                content=block_data.get("content", ""),
                language=block_data.get("language", "text"),
                file_path=block_data.get("file_path"),
                hash=block_data.get("hash", ""),
                purpose=block_data.get("purpose", "")
            )
            
            # Increment ID counter if used
            if block.id == f"block_{self.next_block_id}":
                self.next_block_id += 1
                
            # Analyze block
            analysis = self.code_analyzer.analyze_block(block.content, block.language)
            
            # Set analyzed components
            block.functions = analysis.get("functions", [])
            block.classes = analysis.get("classes", [])
            block.imports = analysis.get("imports", [])
            block.variables = analysis.get("variables", [])
            block.api_surface = analysis.get("api_surface", {})
            
            if not block.purpose and "purpose" in analysis:
                block.purpose = analysis["purpose"]
                
            # Add to graph
            self.dependency_graph.add_block(block)
        
        # Infer dependencies between blocks
        self._infer_block_dependencies()
    
    def _infer_block_dependencies(self) -> None:
        """Infer dependencies between code blocks based on analysis"""
        # Group blocks by language
        blocks_by_language = {}
        for block_id, block in self.dependency_graph.blocks.items():
            if block.language not in blocks_by_language:
                blocks_by_language[block.language] = []
            blocks_by_language[block.language].append(block)
        
        # For each language, infer dependencies
        for language, blocks in blocks_by_language.items():
            self._infer_dependencies_for_language(language, blocks)
    
    def _infer_dependencies_for_language(self, language: str, blocks: List[CodeBlock]) -> None:
        """Infer dependencies for blocks of a specific language"""
        if language == "python":
            self._infer_python_dependencies(blocks)
        elif language in ["javascript", "typescript"]:
            self._infer_js_dependencies(blocks)
    
    def _infer_python_dependencies(self, blocks: List[CodeBlock]) -> None:
        """Infer dependencies between Python code blocks"""
        # Map of class/function names to block IDs
        name_to_block = {}
        
        # Build map of exported names
        for block in blocks:
            for func in block.functions:
                name_to_block[func["name"]] = block.id
                
            for cls in block.classes:
                name_to_block[cls["name"]] = block.id
        
        # Look for imports and function calls that match other blocks
        for block in blocks:
            # Check imports for local modules
            for imp in block.imports:
                module = imp.get("module", "")
                if "." in module:
                    # Extract last part of module path
                    module_parts = module.split(".")
                    local_module = module_parts[-1]
                    
                    # Check if this matches a file path
                    for other_block in blocks:
                        if other_block.file_path:
                            file_name = os.path.basename(other_block.file_path)
                            name_without_ext = os.path.splitext(file_name)[0]
                            
                            if name_without_ext == local_module:
                                self.dependency_graph.add_dependency(block.id, other_block.id)
            
            # Check function calls
            analysis = self.code_analyzer.analyze_block(block.content, "python")
            function_calls = analysis.get("function_calls", [])
            
            for call in function_calls:
                func_name = call.get("function", "")
                if func_name in name_to_block and name_to_block[func_name] != block.id:
                    self.dependency_graph.add_dependency(block.id, name_to_block[func_name])
    
    def _infer_js_dependencies(self, blocks: List[CodeBlock]) -> None:
        """Infer dependencies between JavaScript/TypeScript code blocks"""
        # Map of exported names to block IDs
        name_to_block = {}
        
        # Build map of exported names
        for block in blocks:
            for func in block.functions:
                name_to_block[func["name"]] = block.id
                
            for cls in block.classes:
                name_to_block[cls["name"]] = block.id
        
        # Look for imports and function calls that match other blocks
        for block in blocks:
            # Check imports
            for imp in block.imports:
                module = imp.get("module", "")
                
                # Check relative imports
                if module.startswith("./") or module.startswith("../"):
                    # Normalize path
                    path_parts = module.split("/")
                    module_name = path_parts[-1]
                    
                    # Check if this matches a file path
                    for other_block in blocks:
                        if other_block.file_path:
                            file_name = os.path.basename(other_block.file_path)
                            name_without_ext = os.path.splitext(file_name)[0]
                            
                            if name_without_ext == module_name:
                                self.dependency_graph.add_dependency(block.id, other_block.id)
            
            # Check function calls
            analysis = self.code_analyzer.analyze_block(block.content, "javascript")
            function_calls = analysis.get("function_calls", [])
            
            for call in function_calls:
                func_name = call.get("function", "")
                if func_name in name_to_block and name_to_block[func_name] != block.id:
                    self.dependency_graph.add_dependency(block.id, name_to_block[func_name])
    
    def _handle_generation_request(
        self,
        query: str,
        code_request: Dict[str, Any],
        existing_code: List[Dict[str, Any]],
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle code generation request"""
        # Extract details from request
        language = code_request["details"].get("language", "python")
        purpose = code_request["details"].get("purpose", "")
        
        # Determine dependencies based on query context
        dependencies = self._identify_relevant_dependencies(query, language)
        
        # Extract requirements from context and project vision
        requirements = self._extract_requirements(query, context_report, evolution_report, project_vision)
        
        # Generate code
        generation_result = self.code_generator.generate_code(
            language=language,
            purpose=purpose,
            dependencies=dependencies,
            requirements=requirements
        )
        
        # Create new code block
        block_id = f"block_{self.next_block_id}"
        self.next_block_id += 1
        
        block = CodeBlock(
            id=block_id,
            content=generation_result["code"],
            language=language,
            purpose=purpose,
            functions=generation_result["analysis"].get("functions", []),
            classes=generation_result["analysis"].get("classes", []),
            imports=generation_result["analysis"].get("imports", []),
            variables=generation_result["analysis"].get("variables", []),
            api_surface=generation_result["analysis"].get("api_surface", {})
        )
        
        # Add to dependency graph
        self.dependency_graph.add_block(block)
        
        # Convert to simpler format for return
        code_block = {
            "id": block.id,
            "language": block.language,
            "content": block.content,
            "hash": block.hash,
            "purpose": block.purpose
        }
        
        # Create dependency visualization
        dependency_diagram = self.dependency_graph.visualize_dependencies(block.id)
        
        return {
            "request_type": "generate",
            "code_blocks": [code_block],
            "explanation": self._generate_explanation(block, dependencies, requirements),
            "dependency_graph": dependency_diagram,
            "details": code_request["details"]
        }
    
    def _handle_modification_request(
        self,
        query: str,
        code_request: Dict[str, Any],
        existing_code: List[Dict[str, Any]],
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle code modification request"""
        # Extract details from request
        modification = code_request["details"].get("modification", "")
        
        # Find the most relevant code block to modify
        block_to_modify = self._identify_block_to_modify(query, existing_code)
        
        if not block_to_modify:
            # If no relevant block found, treat as generation
            return self._handle_generation_request(
                query, 
                code_request, 
                existing_code, 
                context_report, 
                evolution_report,
                project_vision
            )
        
        # Get dependencies of the block
        block_id = block_to_modify.get("id")
        dependency_ids = self.dependency_graph.get_dependencies(block_id)
        dependencies = [self.dependency_graph.get_block(dep_id) for dep_id in dependency_ids]
        dependencies = [dep for dep in dependencies if dep]  # Filter out None
        
        # Extract requirements
        requirements = self._extract_requirements(query, context_report, evolution_report, project_vision)
        
        # Get the original block from dependency graph
        original_block = self.dependency_graph.get_block(block_id)
        
        if not original_block:
            # If block not in graph, create it
            original_block = CodeBlock(
                id=block_id,
                content=block_to_modify.get("content", ""),
                language=block_to_modify.get("language", "python"),
                purpose=block_to_modify.get("purpose", "")
            )
            self.dependency_graph.add_block(original_block)
        
        # Modify code
        modification_result = self.code_generator.modify_code(
            original_block=original_block,
            modification_description=modification,
            dependencies=dependencies,
            requirements=requirements
        )
        
        # Check impact of modification
        impact = self.dependency_graph.check_update_impact(block_id, modification_result["code"])
        
        # Update block in dependency graph
        new_block, affected_dependent_ids = self.dependency_graph.update_block(block_id, modification_result["code"])
        
        # Analyze affected dependents
        affected_dependents = []
        for dep_id in affected_dependent_ids:
            dep_block = self.dependency_graph.get_block(dep_id)
            if dep_block:
                affected_dependents.append({
                    "id": dep_id,
                    "file_path": dep_block.file_path,
                    "purpose": dep_block.purpose
                })
        
        # Convert to simpler format for return
        code_block = {
            "id": new_block.id,
            "language": new_block.language,
            "content": new_block.content,
            "hash": new_block.hash,
            "purpose": new_block.purpose
        }
        
        # Create dependency visualization
        dependency_diagram = self.dependency_graph.visualize_dependencies(new_block.id)
        
        return {
            "request_type": "modify",
            "code_blocks": [code_block],
            "original_block": {
                "id": original_block.id,
                "content": original_block.content
            },
            "affected_dependents": affected_dependents,
            "explanation": self._generate_modification_explanation(
                original_block, new_block, modification, dependencies, requirements
            ),
            "dependency_graph": dependency_diagram,
            "details": code_request["details"]
        }
    
    def _handle_analysis_request(
        self,
        query: str,
        code_request: Dict[str, Any],
        existing_code: List[Dict[str, Any]],
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle code analysis request"""
        # Find the most relevant code block to analyze
        block_to_analyze = self._identify_block_to_modify(query, existing_code)
        
        if not block_to_analyze:
            return {
                "request_type": "analyze",
                "explanation": "No relevant code found to analyze.",
                "details": code_request["details"]
            }
        
        # Get the block from dependency graph
        block_id = block_to_analyze.get("id")
        block = self.dependency_graph.get_block(block_id)
        
        if not block:
            # If not in graph, create temporary block for analysis
            block = CodeBlock(
                id=block_id,
                content=block_to_analyze.get("content", ""),
                language=block_to_analyze.get("language", "python"),
                purpose=block_to_analyze.get("purpose", "")
            )
        
        # Run in-depth analysis using LLM
        analysis_prompt = f"""
        Analyze this {block.language} code:
        
        ```{block.language}
        {block.content}
        ```
        
        Please provide a detailed analysis including:
        1. Overview of what the code does
        2. Key functions and classes
        3. Code quality assessment (structure, readability, error handling)
        4. Potential bugs or issues
        5. Suggestions for improvement
        
        Focus on providing valuable insights about the code's structure, quality, and potential improvements.
        """
        
        analysis = self.llm.predict(analysis_prompt)
        
        # Get dependencies and dependents
        dependency_ids = self.dependency_graph.get_dependencies(block_id)
        dependencies = [self.dependency_graph.get_block(dep_id) for dep_id in dependency_ids]
        dependencies = [dep for dep in dependencies if dep]  # Filter out None
        
        dependent_ids = self.dependency_graph.get_dependents(block_id)
        dependents = [self.dependency_graph.get_block(dep_id) for dep_id in dependent_ids]
        dependents = [dep for dep in dependents if dep]  # Filter out None
        
        # Create dependency visualization
        dependency_diagram = self.dependency_graph.visualize_dependencies(block_id)
        
        return {
            "request_type": "analyze",
            "original_block": {
                "id": block_id,
                "language": block.language,
                "content": block.content,
                "purpose": block.purpose
            },
            "analysis": analysis,
            "dependencies": [{"id": dep.id, "purpose": dep.purpose} for dep in dependencies],
            "dependents": [{"id": dep.id, "purpose": dep.purpose} for dep in dependents],
            "dependency_graph": dependency_diagram,
            "details": code_request["details"]
        }
    
    def _identify_relevant_dependencies(self, query: str, language: str) -> List[CodeBlock]:
        """Identify relevant dependencies based on query context"""
        # Get blocks of the same language
        all_blocks = self.dependency_graph.get_blocks_by_language(language)
        
        if not all_blocks:
            return []
            
        # Use query to rank relevance (simple keyword matching)
        ranked_blocks = []
        for block in all_blocks:
            # Skip blocks with no content
            if not block.content:
                continue
                
            # Calculate relevance score
            relevance = 0
            
            # Check purpose match
            if block.purpose and any(word in block.purpose.lower() for word in query.lower().split()):
                relevance += 2
                
            # Check functions and classes
            for func in block.functions:
                if any(word in func.get("name", "").lower() for word in query.lower().split()):
                    relevance += 1
                    
            for cls in block.classes:
                if any(word in cls.get("name", "").lower() for word in query.lower().split()):
                    relevance += 1
            
            # Add if relevant
            if relevance > 0:
                ranked_blocks.append((block, relevance))
        
        # Sort by relevance (descending)
        ranked_blocks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top blocks (max 3)
        return [block for block, _ in ranked_blocks[:3]]
    
    def _identify_block_to_modify(self, query: str, existing_code: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify which code block to modify based on query context"""
        if not existing_code:
            return None
            
        # If only one block exists, use that
        if len(existing_code) == 1:
            return existing_code[0]
            
        # Use query to rank relevance (simple keyword matching)
        ranked_blocks = []
        for block in existing_code:
            # Calculate relevance score
            relevance = 0
            content = block.get("content", "")
            language = block.get("language", "")
            purpose = block.get("purpose", "")
            
            # Check language match
            if language and language.lower() in query.lower():
                relevance += 3
                
            # Check purpose match
            if purpose and any(word in purpose.lower() for word in query.lower().split()):
                relevance += 2
                
            # Check content (functions/classes/variables)
            if content:
                lines = content.splitlines()
                for line in lines:
                    # Look for function/class definitions or variables mentioned in query
                    if ("def " in line or "class " in line or "=" in line) and \
                       any(word in line.lower() for word in query.lower().split()):
                        relevance += 1
            
            # Add if relevant
            if relevance > 0:
                ranked_blocks.append((block, relevance))
        
        # Sort by relevance (descending)
        ranked_blocks.sort(key=lambda x: x[1], reverse=True)
        
        # Return most relevant block if any found
        return ranked_blocks[0][0] if ranked_blocks else existing_code[-1]  # Default to most recent
    
    def _extract_requirements(
        self,
        query: str,
        context_report: Dict[str, Any],
        evolution_report: Dict[str, Any],
        project_vision: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract requirements from context and reports"""
        requirements = []
        
        # Extract from query
        query_requirements = []
        requirement_markers = ["must", "should", "need to", "has to", "requirement"]
        
        for marker in requirement_markers:
            if marker in query.lower():
                # Find sentences containing the marker
                sentences = re.split(r'[.!?]', query)
                for sentence in sentences:
                    if marker in sentence.lower():
                        query_requirements.append(sentence.strip())
        
        # Add query requirements
        requirements.extend(query_requirements)
        
        # Extract from context report (missing contexts)
        if "missing_contexts" in context_report:
            for ctx in context_report.get("missing_contexts", []):
                requirements.append(f"Include context: {ctx.get('context', '')}")
        
        # Extract from evolution report (dropped concepts)
        if "dropped_concepts" in evolution_report:
            for concept in evolution_report.get("dropped_concepts", []):
                requirements.append(f"Include concept: {concept.get('concept', '')}")
        
        # Extract from evolution report (continuity suggestions)
        if "continuity_suggestions" in evolution_report:
            for suggestion in evolution_report.get("continuity_suggestions", []):
                requirements.append(f"Ensure continuity: {suggestion.get('suggestion', '')}")
        
        # Extract from project vision
        if project_vision:
            for req in project_vision.get("requirements", []):
                requirements.append(f"Project requirement: {req}")
                
            for constraint in project_vision.get("constraints", []):
                requirements.append(f"Project constraint: {constraint}")
        
        return requirements
    
    def _generate_explanation(
        self,
        block: CodeBlock,
        dependencies: List[CodeBlock],
        requirements: List[str]
    ) -> str:
        """Generate explanation for generated code"""
        # Create explanation prompt
        prompt = f"""
        Explain this {block.language} code you've generated:
        
        ```{block.language}
        {block.content}
        ```
        
        Focus on explaining:
        1. What the code does and how it works
        2. How it addresses the requirements
        3. Key functions and components
        4. Any important design decisions
        
        Requirements that were addressed:
        {chr(10).join([f"- {req}" for req in requirements]) if requirements else "- No specific requirements"}
        
        Keep your explanation clear, informative, and focused on the most important aspects.
        """
        
        explanation = self.llm.predict(prompt)
        return explanation
    
    def _generate_modification_explanation(
        self,
        original_block: CodeBlock,
        new_block: CodeBlock,
        modification: str,
        dependencies: List[CodeBlock],
        requirements: List[str]
    ) -> str:
        """Generate explanation for code modification"""
        # Create explanation prompt
        prompt = f"""
        Explain the changes made to this {original_block.language} code:
        
        Original purpose: {original_block.purpose}
        Modification requested: {modification}
        
        ORIGINAL CODE:
        ```{original_block.language}
        {original_block.content}
        ```
        
        MODIFIED CODE:
        ```{new_block.language}
        {new_block.content}
        ```
        
        Requirements that were addressed:
        {chr(10).join([f"- {req}" for req in requirements]) if requirements else "- No specific requirements"}
        
        Please explain:
        1. What specific changes were made
        2. How these changes address the requested modification
        3. How dependencies are maintained
        4. Any important design decisions or tradeoffs
        
        Keep your explanation clear, informative, and focused on the changes.
        """
        
        explanation = self.llm.predict(prompt)
        return explanation
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks from text"""
        code_blocks = []
        
        # Match markdown code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for idx, (language, code) in enumerate(matches):
            # Generate a hash for the code block for tracking changes
            code_hash = hashlib.md5(code.encode()).hexdigest()
            block_id = f"block_{self.next_block_id}"
            self.next_block_id += 1
            
            code_blocks.append({
                "id": block_id,
                "language": language.strip() if language else "text",
                "content": code.strip(),
                "hash": code_hash
            })
        
        return code_blocks

