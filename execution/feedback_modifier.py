from typing import Dict, Any, Optional, Tuple, List
import logging
import json

logger = logging.getLogger(__name__)

class FeedbackBasedExecutionModifier:
    """
    Modifies Execution Agent behavior based on user feedback
    
    This component:
    1. Translates user preferences into execution parameters
    2. Maintains user-specific execution profiles
    3. Injects parameters directly into code generation
    4. Personalizes code style based on feedback
    """
    
    def __init__(self, user_profile_manager: Optional[Any] = None):
        """
        Initialize execution modifier
        
        Args:
            user_profile_manager: Optional manager for user profiles
        """
        self.user_profile_manager = user_profile_manager
        self.default_style = {
            "indentation": 4,
            "line_length": 80,
            "comment_style": "inline",
            "naming_convention": "snake_case",
            "verbosity": "medium"
        }
        self.default_structure = {
            "modularization": "medium",
            "error_handling": "comprehensive",
            "use_classes": "when_appropriate",
            "functional_style": False
        }
        self.modification_stats = {
            "total_modifications": 0,
            "user_specific_modifications": 0,
            "default_fallbacks": 0
        }
        
    def get_execution_parameters(
        self, 
        user_id: Optional[str], 
        code_request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get execution parameters based on user preferences
        
        Args:
            user_id: User ID for personalized parameters
            code_request: Details of the code generation request
            
        Returns:
            Tuple of (style_preferences, structure_preferences)
        """
        self.modification_stats["total_modifications"] += 1
        
        # If no user ID or profile manager, use defaults
        if not user_id or not self.user_profile_manager:
            self.modification_stats["default_fallbacks"] += 1
            return self._get_default_parameters(code_request)
        
        # Get user profile
        try:
            user_profile = self.user_profile_manager.get_user_profile(user_id)
            
            if not user_profile:
                self.modification_stats["default_fallbacks"] += 1
                return self._get_default_parameters(code_request)
                
            self.modification_stats["user_specific_modifications"] += 1
            
            # Extract code style preferences
            style_prefs = user_profile.get("code_style", self.default_style)
            
            # Extract code structure preferences
            structure_prefs = user_profile.get("code_structure", self.default_structure)
            
            # Merge with defaults to ensure all expected parameters exist
            style_prefs = {**self.default_style, **style_prefs}
            structure_prefs = {**self.default_structure, **structure_prefs}
            
            # Add language-specific preferences if available
            language = code_request.get("language", "").lower()
            if language:
                language_prefs = user_profile.get(f"{language}_preferences", {})
                style_prefs.update(language_prefs.get("style", {}))
                structure_prefs.update(language_prefs.get("structure", {}))
            
            return style_prefs, structure_prefs
            
        except Exception as e:
            logger.warning(f"Error getting user preferences: {e}")
            self.modification_stats["default_fallbacks"] += 1
            return self._get_default_parameters(code_request)
    
    def _get_default_parameters(self, code_request: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get default parameters based on code request"""
        style_prefs = self.default_style.copy()
        structure_prefs = self.default_structure.copy()
        
        # Adjust defaults based on language if known
        language = code_request.get("language", "").lower()
        
        if language == "python":
            style_prefs["indentation"] = 4
            style_prefs["naming_convention"] = "snake_case"
            
        elif language == "javascript" or language == "typescript":
            style_prefs["indentation"] = 2
            style_prefs["naming_convention"] = "camelCase"
            
        elif language == "java":
            style_prefs["naming_convention"] = "camelCase"
            structure_prefs["use_classes"] = "always"
            
        elif language == "ruby":
            style_prefs["naming_convention"] = "snake_case"
            
        elif language == "go":
            style_prefs["naming_convention"] = "camelCase"
            
        # Try to infer appropriate structure from code request
        purpose = code_request.get("purpose", "").lower()
        
        if "script" in purpose or "utility" in purpose:
            structure_prefs["modularization"] = "low"
            structure_prefs["use_classes"] = "rarely"
            
        elif "library" in purpose or "framework" in purpose or "api" in purpose:
            structure_prefs["modularization"] = "high"
            structure_prefs["error_handling"] = "comprehensive"
            
        elif "algorithm" in purpose or "function" in purpose:
            structure_prefs["modularization"] = "medium"
            
        # Check for explicit style requests in purpose
        if "functional" in purpose or "immutable" in purpose:
            structure_prefs["functional_style"] = True
            
        if "object" in purpose and "oriented" in purpose:
            structure_prefs["use_classes"] = "always"
        
        return style_prefs, structure_prefs
    
    def update_preferences_from_feedback(
        self, 
        user_id: str, 
        feedback: str, 
        code_block: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences based on feedback
        
        Args:
            user_id: User ID to update preferences for
            feedback: User feedback text
            code_block: Code block the feedback refers to
            
        Returns:
            Success flag
        """
        if not user_id or not self.user_profile_manager:
            return False
            
        # Extract style preferences from feedback
        style_prefs = self._extract_style_preferences(feedback)
        structure_prefs = self._extract_structure_preferences(feedback)
        
        # If no preferences extracted, nothing to update
        if not style_prefs and not structure_prefs:
            return False
            
        try:
            # Get current profile
            profile = self.user_profile_manager.get_user_profile(user_id)
            
            if not profile:
                profile = {"user_id": user_id}
                
            # Update style preferences
            if "code_style" not in profile:
                profile["code_style"] = {}
                
            profile["code_style"].update(style_prefs)
            
            # Update structure preferences
            if "code_structure" not in profile:
                profile["code_structure"] = {}
                
            profile["code_structure"].update(structure_prefs)
            
            # Update language-specific preferences if language is known
            language = code_block.get("language", "").lower()
            if language:
                if f"{language}_preferences" not in profile:
                    profile[f"{language}_preferences"] = {"style": {}, "structure": {}}
                    
                profile[f"{language}_preferences"]["style"].update(style_prefs)
                profile[f"{language}_preferences"]["structure"].update(structure_prefs)
            
            # Save updated profile
            self.user_profile_manager.update_user_profile(user_id, profile)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error updating preferences from feedback: {e}")
            return False
    
    def _extract_style_preferences(self, feedback: str) -> Dict[str, Any]:
        """Extract code style preferences from feedback text"""
        style_prefs = {}
        feedback_lower = feedback.lower()
        
        # Check for indentation preferences
        if "indent" in feedback_lower or "indentation" in feedback_lower or "spaces" in feedback_lower:
            # Check for specific indentation width
            indent_match = re.search(r'(\d+)\s*(?:space|tab|indent)', feedback_lower)
            if indent_match:
                style_prefs["indentation"] = int(indent_match.group(1))
            elif "tab" in feedback_lower:
                style_prefs["indentation"] = "tab"
        
        # Check for line length preferences
        if "line length" in feedback_lower or "line width" in feedback_lower:
            length_match = re.search(r'(\d+)\s*(?:character|char|column)', feedback_lower)
            if length_match:
                style_prefs["line_length"] = int(length_match.group(1))
        
        # Check for comment style preferences
        if "comment" in feedback_lower:
            if "more comment" in feedback_lower or "add comment" in feedback_lower:
                style_prefs["comment_style"] = "comprehensive"
            elif "less comment" in feedback_lower or "fewer comment" in feedback_lower:
                style_prefs["comment_style"] = "minimal"
            elif "doc" in feedback_lower and "string" in feedback_lower:
                style_prefs["comment_style"] = "docstring"
            elif "inline" in feedback_lower:
                style_prefs["comment_style"] = "inline"
        
        # Check for naming convention preferences
        if "naming" in feedback_lower or "name" in feedback_lower:
            if "camel case" in feedback_lower or "camelCase" in feedback_lower:
                style_prefs["naming_convention"] = "camelCase"
            elif "snake case" in feedback_lower or "snake_case" in feedback_lower:
                style_prefs["naming_convention"] = "snake_case"
            elif "pascal case" in feedback_lower or "PascalCase" in feedback_lower:
                style_prefs["naming_convention"] = "PascalCase"
            elif "kebab case" in feedback_lower or "kebab-case" in feedback_lower:
                style_prefs["naming_convention"] = "kebab-case"
        
        # Check for verbosity preferences
        if "verbos" in feedback_lower or "concise" in feedback_lower or "terse" in feedback_lower:
            if "more verbose" in feedback_lower or "less concise" in feedback_lower:
                style_prefs["verbosity"] = "high"
            elif "less verbose" in feedback_lower or "more concise" in feedback_lower or "terse" in feedback_lower:
                style_prefs["verbosity"] = "low"
        
        return style_prefs
    
    def _extract_structure_preferences(self, feedback: str) -> Dict[str, Any]:
        """Extract code structure preferences from feedback text"""
        structure_prefs = {}
        feedback_lower = feedback.lower()
        
        # Check for modularization preferences
        if "modulari" in feedback_lower or "split" in feedback_lower or "separate" in feedback_lower:
            if "more modular" in feedback_lower or "increase modularity" in feedback_lower:
                structure_prefs["modularization"] = "high"
            elif "less modular" in feedback_lower or "decrease modularity" in feedback_lower:
                structure_prefs["modularization"] = "low"
        
        # Check for error handling preferences
        if "error" in feedback_lower or "exception" in feedback_lower:
            if "more error" in feedback_lower or "better error" in feedback_lower:
                structure_prefs["error_handling"] = "comprehensive"
            elif "less error" in feedback_lower or "fewer exception" in feedback_lower:
                structure_prefs["error_handling"] = "minimal"
        
        # Check for class usage preferences
        if "class" in feedback_lower or "object" in feedback_lower:
            if "more class" in feedback_lower or "object-oriented" in feedback_lower:
                structure_prefs["use_classes"] = "always"
            elif "less class" in feedback_lower or "fewer class" in feedback_lower:
                structure_prefs["use_classes"] = "rarely"
        
        # Check for functional style preferences
        if "function" in feedback_lower:
            if "functional style" in feedback_lower or "functional programming" in feedback_lower:
                structure_prefs["functional_style"] = True
            elif "less functional" in feedback_lower:
                structure_prefs["functional_style"] = False
        
        return structure_prefs
    
    def get_stats(self) -> Dict[str, int]:
        """Get modification statistics"""
        return self.modification_stats.copy()
