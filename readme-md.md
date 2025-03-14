# WiseFlow: Advanced Context-Aware AI Assistant Framework

WiseFlow is a sophisticated AI assistant framework designed with a strong focus on maintaining context awareness, ensuring logical coherence, generating and analyzing code, and continuously self-improving.

## Features

- **Deep Context Awareness**: Never drops important information from context
- **Logical Coherence Validation**: Ensures responses remain logically consistent
- **Concept Tracking**: Maintains and evolves concepts throughout conversations
- **Code Generation & Analysis**: Generates, tests, and analyzes code with deep semantic understanding
- **Self-Improvement Mechanisms**: Continuously optimizes prompts and monitors system health

## Architecture

WiseFlow uses a modular architecture with specialized components:

- **Core**: Central orchestration and component management
- **Context**: Tracking conversation context and concepts
- **Validation**: Ensuring logical coherence and consistency
- **Execution**: Code generation and execution
- **Optimization**: Self-improvement and quality monitoring

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/wiseflow.git
   cd wiseflow
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

4. Create a `.env` file with your configuration:
   ```
   # API Keys for Language Models
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here

   # Database Configuration
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=wiseflow
   DB_USER=your_postgres_user
   DB_PASSWORD=your_postgres_password
   ```

### Running the System

```
python -m core.bootstrap
```

## Development

### Running Tests

```
pytest
```

### Project Structure

- `context/`: Context tracking and management
- `core/`: Core system components
- `execution/`: Code generation and execution
- `optimization/`: Self-improvement mechanisms
- `validation/`: Logical validation components
- `utils/`: Utility functions and tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.
