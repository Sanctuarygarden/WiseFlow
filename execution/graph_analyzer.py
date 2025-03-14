class GraphOfThoughtCodeAnalyzer:
    """
    Enhanced code analyzer with Graph-of-Thought reasoning
    
    This component:
    1. Builds a comprehensive graph representation of code elements and relationships
    2. Uses graph analysis to identify hidden dependencies and relationships
    3. Reasons about code structure and function systematically
    4. Provides deeper semantic understanding of code purpose and behavior
    """
    
    def __init__(self, llm, dependency_graph):
        self.llm = llm
        self.dependency_graph = dependency_graph
        self.semantic_graph = nx.DiGraph()  # Using networkx for graph analysis
        
    def analyze_codebase(self, blocks):
        """Perform comprehensive analysis of entire codebase"""
        # Step 1: Build initial semantic graph from code blocks
        self._build_semantic_graph(blocks)
        
        # Step 2: Enhance graph with inferred relationships
        self._enhance_graph_with_got()
        
        # Step 3: Perform graph analysis
        analysis = self._analyze_semantic_graph()
        
        return analysis
    
    def _build_semantic_graph(self, blocks):
        """Build initial semantic graph from code blocks"""
        # Clear existing graph
        self.semantic_graph.clear()
        
        # Add nodes for each code block
        for block in blocks:
            self.semantic_graph.add_node(
                block.id,
                type="code_block",
                language=block.language,
                file_path=block.file_path,
                content=block.content
            )
            
            # Add function nodes
            for func in block.functions:
                func_id = f"{block.id}_func_{func['name']}"
                self.semantic_graph.add_node(
                    func_id,
                    type="function",
                    name=func["name"],
                    block_id=block.id,
                    params=func.get("params", [])
                )
                
                # Add edge from block to function
                self.semantic_graph.add_edge(block.id, func_id, type="contains")
            
            # Add class nodes
            for cls in block.classes:
                class_id = f"{block.id}_class_{cls['name']}"
                self.semantic_graph.add_node(
                    class_id,
                    type="class",
                    name=cls["name"],
                    block_id=block.id,
                    methods=cls.get("methods", [])
                )
                
                # Add edge from block to class
                self.semantic_graph.add_edge(block.id, class_id, type="contains")
        
        # Add edges for dependencies
        for block in blocks:
            for dep_id in block.dependencies:
                if self.semantic_graph.has_node(dep_id):
                    self.semantic_graph.add_edge(block.id, dep_id, type="depends_on")
    
    def _enhance_graph_with_got(self):
        """Enhance semantic graph using Graph-of-Thought reasoning"""
        # Only process if we have an LLM
        if not self.llm:
            return
            
        # Step 1: Identify potential hidden relationships
        potential_relationships = self._identify_potential_relationships()
        
        # Step 2: Validate relationships with LLM
        validated_relationships = self._validate_relationships(potential_relationships)
        
        # Step 3: Add validated relationships to graph
        for relationship in validated_relationships:
            source_id = relationship["source_id"]
            target_id = relationship["target_id"]
            rel_type = relationship["relationship_type"]
            
            # Add edge if nodes exist
            if self.semantic_graph.has_node(source_id) and self.semantic_graph.has_node(target_id):
                self.semantic_graph.add_edge(
                    source_id, 
                    target_id, 
                    type=rel_type,
                    confidence=relationship.get("confidence", 0.5),
                    description=relationship.get("description", "")
                )
    
    def _identify_potential_relationships(self):
        """Identify potential hidden relationships in the code"""
        potential_relationships = []
        
        # Get all function and class nodes
        function_nodes = [
            (node_id, data) for node_id, data in self.semantic_graph.nodes(data=True)
            if data.get("type") == "function"
        ]
        
        class_nodes = [
            (node_id, data) for node_id, data in self.semantic_graph.nodes(data=True)
            if data.get("type") == "class"
        ]
        
        # Look for potential relationships between functions
        for func1_id, func1_data in function_nodes:
            func1_block_id = func1_data.get("block_id")
            func1_block = self.dependency_graph.get_block(func1_block_id)
            
            if not func1_block:
                continue
                
            func1_name = func1_data.get("name", "")
            
            for func2_id, func2_data in function_nodes:
                # Skip self-relationships
                if func1_id == func2_id:
                    continue
                    
                func2_block_id = func2_data.get("block_id")
                func2_block = self.dependency_graph.get_block(func2_block_id)
                
                if not func2_block:
                    continue
                    
                func2_name = func2_data.get("name", "")
                
                # Check if function names indicate a relationship
                if self._names_suggest_relationship(func1_name, func2_name):
                    potential_relationships.append({
                        "source_id": func1_id,
                        "target_id": func2_id,
                        "source_name": func1_name,
                        "target_name": func2_name,
                        "relationship_type": "semantically_related",
                        "source_block_id": func1_block_id,
                        "target_block_id": func2_block_id
                    })
        
        # Look for relationships between class methods and functions
        for class_id, class_data in class_nodes:
            class_block_id = class_data.get("block_id")
            class_name = class_data.get("name", "")
            
            for func_id, func_data in function_nodes:
                func_block_id = func_data.get("block_id")
                func_name = func_data.get("name", "")
                
                # Check if function might be a factory or utility for the class
                if (class_name.lower() in func_name.lower() or 
                    func_name.lower().endswith(f"_{class_name.lower()}")):
                    potential_relationships.append({
                        "source_id": func_id,
                        "target_id": class_id,
                        "source_name": func_name,
                        "target_name": class_name,
                        "relationship_type": "related_to_class",
                        "source_block_id": func_block_id,
                        "target_block_id": class_block_id
                    })
        
        return potential_relationships
    
    def _names_suggest_relationship(self, name1, name2):
        """Check if two names suggest a semantic relationship"""
        name1_parts = self._split_name(name1)
        name2_parts = self._split_name(name2)
        
        # Check for shared word stems
        common_parts = set(name1_parts) & set(name2_parts)
        if common_parts and len(common_parts) / max(len(name1_parts), len(name2_parts)) > 0.3:
            return True
            
        # Check for common prefixes/suffixes
        prefixes = ["get", "set", "create", "build", "parse", "format", "convert", "transform"]
        
        for prefix in prefixes:
            if (name1.startswith(prefix) and name2.startswith(prefix) and
                self._stem_compare(name1[len(prefix):], name2[len(prefix):])):
                return True
        
        return False
    
    def _split_name(self, name):
        """Split a name into its component parts"""
        # Handle camelCase
        parts = []
        current_part = ""
        
        for char in name:
            if char.isupper() and current_part:
                parts.append(current_part.lower())
                current_part = char.lower()
            else:
                current_part += char.lower()
                
        if current_part:
            parts.append(current_part.lower())
            
        # Handle snake_case
        result = []
        for part in parts:
            result.extend(part.split('_'))
            
        return [p for p in result if p]
    
    def _stem_compare(self, str1, str2):
        """Compare string stems for similarity"""
        # Simple stem comparison
        str1 = str1.lower()
        str2 = str2.lower()
        
        # Check if one is contained in the other
        if str1 in str2 or str2 in str1:
            return True
            
        # Check for edit distance
        if len(str1) > 3 and len(str2) > 3:
            # Simple Levenshtein distance would be calculated here
            # For simplicity, we'll use a substring check
            for i in range(3, min(len(str1), len(str2)) + 1):
                if str1[:i] == str2[:i]:
                    return True
        
        return False
    
    def _validate_relationships(self, potential_relationships):
        """Validate potential relationships using LLM"""
        validated_relationships = []
        
        # If no LLM or no relationships, return empty list
        if not self.llm or not potential_relationships:
            return validated_relationships
            
        # Process relationships in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(potential_relationships), batch_size):
            batch = potential_relationships[i:i+batch_size]
            
            # Create validation prompt
            relationships_text = []
            
            for j, rel in enumerate(batch):
                source_block = self.dependency_graph.get_block(rel["source_block_id"])
                target_block = self.dependency_graph.get_block(rel["target_block_id"])
                
                if not source_block or not target_block:
                    continue
                    
                relationships_text.append(f"Relationship {j+1}:")
                relationships_text.append(f"Source: {rel['source_name']} ({rel['source_id']})")
                relationships_text.append(f"Target: {rel['target_name']} ({rel['target_id']})")
                relationships_text.append(f"Suggested Relationship: {rel['relationship_type']}")
                relationships_text.append("\nSource code snippet:")
                
                # Extract relevant function/class from source block
                source_snippet = self._extract_element_snippet(source_block.content, rel["source_name"])
                relationships_text.append(f"```{source_block.language}\n{source_snippet}\n```")
                
                relationships_text.append("\nTarget code snippet:")
                target_snippet = self._extract_element_snippet(target_block.content, rel["target_name"])
                relationships_text.append(f"```{target_block.language}\n{target_snippet}\n```\n")
            
            prompt = f"""
            Validate these potential code relationships using Graph-of-Thought reasoning:
            
            {chr(10).join(relationships_text)}
            
            For each relationship, determine:
            1. Is this relationship valid? (yes/no)
            2. If valid, what type of relationship is it? (e.g., "calls", "extends", "implements", "semantically_related", etc.)
            3. Confidence in your assessment (0.0-1.0)
            4. Brief description of the relationship
            
            Return your analysis as a JSON array where each object represents one relationship.
            """
            
            try:
                response = self.llm.predict(prompt)
                
                # Parse JSON response
                import re
                import json
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                
                if json_match:
                    validation_results = json.loads(json_match.group(0))
                    
                    # Process validation results
                    for j, result in enumerate(validation_results):
                        if j < len(batch) and result.get("valid", False):
                            relationship = batch[j].copy()
                            relationship["relationship_type"] = result.get("relationship_type", relationship["relationship_type"])
                            relationship["confidence"] = result.get("confidence", 0.5)
                            relationship["description"] = result.get("description", "")
                            
                            validated_relationships.append(relationship)
            except Exception as e:
                logger.warning(f"Relationship validation failed: {e}")
        
        return validated_relationships
    
    def _extract_element_snippet(self, content, element_name):
        """Extract a code snippet for a specific element"""
        import re
        
        # Try to find the element by name
        # This is a simple implementation that would be enhanced in practice
        lines = content.split('\n')
        
        # Common patterns for function, method, and class definitions
        patterns = [
            fr"def\s+{re.escape(element_name)}\s*\(", # Python function
            fr"class\s+{re.escape(element_name)}\s*[(:)]", # Python/JS class
            fr"function\s+{re.escape(element_name)}\s*\(", # JS function
            fr"const\s+{re.escape(element_name)}\s*=\s*function", # JS function expression
            fr"const\s+{re.escape(element_name)}\s*=\s*\(" # JS arrow function
        ]
        
        # Find the start line
        start_line = -1
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in patterns):
                start_line = i
                break
        
        if start_line == -1:
            # Element not found, return a small portion of the code
            return "\n".join(lines[:min(10, len(lines))])
        
        # Estimate the end of the element (would be enhanced in practice)
        # This is a simplistic approach that tries to find the end of a block
        end_line = start_line + 1
        indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        while end_line < len(lines):
            if lines[end_line].strip() and len(lines[end_line]) - len(lines[end_line].lstrip()) <= indent:
                # Found a line with same or lower indentation
                break
            end_line += 1
        
        # Include a few lines before and after for context
        context_before = max(0, start_line - 2)
        context_after = min(len(lines), end_line + 2)
        
        return "\n".join(lines[context_before:context_after])
    
    def _analyze_semantic_graph(self):
        """Analyze the semantic graph for insights"""
        # Simple graph analysis
        analysis = {
            "node_count": self.semantic_graph.number_of_nodes(),
            "edge_count": self.semantic_graph.number_of_edges(),
            "code_blocks": 0,
            "functions": 0,
            "classes": 0,
            "centrality": {},
            "communities": [],
            "insights": []
        }
        
        # Count node types
        for _, data in self.semantic_graph.nodes(data=True):
            node_type = data.get("type", "")
            if node_type == "code_block":
                analysis["code_blocks"] += 1
            elif node_type == "function":
                analysis["functions"] += 1
            elif node_type == "class":
                analysis["classes"] += 1
        
        # Calculate centrality (importance) of nodes
        try:
            import networkx as nx
            centrality = nx.degree_centrality(self.semantic_graph)
            analysis["centrality"] = {
                node: centrality[node]
                for node in sorted(centrality, key=centrality.get, reverse=True)[:10]  # Top 10
            }
            
            # Try to detect communities (related components)
            try:
                communities = list(nx.community.greedy_modularity_communities(self.semantic_graph.to_undirected()))
                
                # Convert communities to lists of node IDs
                analysis["communities"] = [list(community) for community in communities]
            except:
                # Community detection might fail on some graphs
                pass
        except:
            # Centrality calculation might fail on some graphs
            pass
        
        # Add graph-based insights
        analysis["insights"] = self._generate_graph_insights()
        
        return analysis
    
    def _generate_graph_insights(self):
        """Generate insights from graph analysis"""
        insights = []
        
        # Find highly connected components
        try:
            import networkx as nx
            centrality = nx.degree_centrality(self.semantic_graph)
            
            # Find nodes with high centrality
            high_centrality = [
                (node, score) for node, score in centrality.items()
                if score > 0.5  # Arbitrary threshold
            ]
            
            for node, score in high_centrality:
                node_data = self.semantic_graph.nodes[node]
                node_type = node_data.get("type", "")
                
                if node_type == "function":
                    insights.append({
                        "type": "high_centrality_function",
                        "element": node_data.get("name", ""),
                        "score": score,
                        "suggestion": "This function is highly connected and may be a critical component."
                    })
                elif node_type == "class":
                    insights.append({
                        "type": "high_centrality_class",
                        "element": node_data.get("name", ""),
                        "score": score,
                        "suggestion": "This class is highly connected and may be a core component."
                    })
        except:
            pass
        
        # Find potential refactoring opportunities
        # Functions with similar names but in different modules
        function_nodes = [
            (node_id, data) for node_id, data in self.semantic_graph.nodes(data=True)
            if data.get("type") == "function"
        ]
        
        function_by_name = {}
        for node_id, data in function_nodes:
            func_name = data.get("name", "")
            if func_name not in function_by_name:
                function_by_name[func_name] = []
            function_by_name[func_name].append((node_id, data))
        
        # Find functions with similar names
        for name, funcs in function_by_name.items():
            if len(funcs) > 1:
                # Check if they're in different blocks
                blocks = set(data.get("block_id") for node_id, data in funcs)
                if len(blocks) > 1:
                    insights.append({
                        "type": "similar_functions",
                        "elements": [data.get("name", "") for node_id, data in funcs],
                        "blocks": list(blocks),
                        "suggestion": f"Multiple functions named '{name}' exist in different modules. Consider consolidation."
                    })
        
        return insights
    
    def visualize_semantic_graph(self):
        """Generate a Mermaid visualization of the semantic graph"""
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
        
        # Add nodes
        for node_id, data in self.semantic_graph.nodes(data=True):
            node_type = data.get("type", "")
            
            if node_type == "code_block":
                label = data.get("file_path", node_id)
                if label == node_id and "language" in data:
                    label = f"{label} ({data['language']})"
                add_node(node_id, label, "codeBlock")
                
            elif node_type == "function":
                name = data.get("name", "")
                params = data.get("params", [])
                param_str = ", ".join([p.get("name", "") for p in params])
                label = f"{name}({param_str})"
                add_node(node_id, label, "function")
                
            elif node_type == "class":
                name = data.get("name", "")
                methods = data.get("methods", [])
                add_node(node_id, f"Class: {name}", "class")
        
        # Add edges
        for source, target, data in self.semantic_graph.edges(data=True):
            edge_type = data.get("type", "")
            
            if edge_type == "contains":
                # No label for contains relationships
                add_edge(source, target)
                
            else:
                # Use edge type as label
                add_edge(source, target, edge_type)
        
        # Add styling
        mermaid.append("    classDef codeBlock fill:#f9f,stroke:#333,stroke-width:1px")
        mermaid.append("    classDef function fill:#9cf,stroke:#333")
        mermaid.append("    classDef class fill:#fc9,stroke:#333")
        
        return "\n".join(mermaid)
    
    def analyze_with_got(self, query, code_blocks):
        """Analyze codebase with Graph-of-Thought reasoning based on query"""
        # Step 1: Build semantic graph
        self._build_semantic_graph(code_blocks)
        
        # Step 2: Generate Graph-of-Thought reasoning
        got_analysis = self._generate_got_analysis(query, code_blocks)
        
        return got_analysis
    
    def _generate_got_analysis(self, query, code_blocks):
        """Generate Graph-of-Thought reasoning about code"""
        if not self.llm:
            return {"reasoning": "LLM not available for Graph-of-Thought reasoning"}
            
        # Format code blocks for the prompt
        blocks_text = []
        for block in code_blocks[:3]:  # Limit to 3 blocks to avoid token limits
            blocks_text.append(f"Block: {block.id}")
            blocks_text.append(f"Language: {block.language}")
            blocks_text.append(f"File: {block.file_path or 'unnamed'}")
            blocks_text.append(f"```{block.language}\n{block.content}\n```\n")
        
        # Create Graph-of-Thought prompt
        prompt = f"""
        Analyze this code using Graph-of-Thought reasoning to answer the query: {query}
        
        CODE BLOCKS:
        {chr(10).join(blocks_text)}
        
        Graph-of-Thought reasoning involves:
        1. Breaking down the code into components (nodes)
        2. Identifying relationships between components (edges)
        3. Analyzing how information and control flows through the graph
        4. Drawing insights based on the graph structure
        
        Produce a step-by-step analysis showing your graph-based reasoning.
        Structure your response with these sections:
        1. Graph Construction (how you mentally model the code structure)
        2. Component Analysis (key functions, classes, and their roles)
        3. Relationship Analysis (how components interact)
        4. Information Flow (how data moves through the system)
        5. Conclusions (answers to the specific query)
        """
        
        try:
            got_reasoning = self.llm.predict(prompt)
            
            return {
                "reasoning": got_reasoning,
                "query": query
            }
        except Exception as e:
            logger.warning(f"Graph-of-Thought analysis failed: {e}")
            return {
                "reasoning": f"Graph-of-Thought analysis failed: {str(e)}",
                "query": query
            }

# Note: This implementation assumes availability of NetworkX (nx) for graph operations
# If not available, the import and graph operations would need to be modified
import networkx as nx

