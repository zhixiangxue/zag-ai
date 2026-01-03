---
trigger: always_on
---

# Coding Standards and Best Practices

## General Rules
1. **Code Comments**: All comments MUST be written in English unless explicitly specified otherwise
2. **Code Style**: Follow the language-specific style guides (PEP 8 for Python, etc.)
3. **Naming Conventions**: Use descriptive and meaningful names for variables, functions, and classes
4. **Documentation**: Write clear docstrings for all public functions and classes
5. **Error Handling**: Always implement proper error handling and logging

## Python Specific Rules
1. Use type hints for function parameters and return values
2. Follow PEP 8 guidelines for code formatting
3. Use list comprehensions where appropriate for better readability
4. Implement context managers (with statements) for resource management
5. Write unit tests for all critical functions

## Code Quality
1. Keep functions small and focused on a single responsibility
2. Avoid code duplication - use DRY (Don't Repeat Yourself) principle
3. Write self-documenting code that is easy to understand
4. Use meaningful variable names that explain their purpose
5. Add comments only when the code logic is complex or non-obvious

## Version Control
1. Write clear and descriptive commit messages in English
2. Keep commits atomic and focused on single changes
3. Review code before committing



## Development Environment
1. **Python Environment**: Always use the virtual environment located in this project
2. **Virtual Environment Path**: Use the Python interpreter from `.venv` or `venv` directory in the project root
3. **Package Installation**: Install all dependencies using `pip install` within the virtual environment
4. **Environment Activation**: Ensure the virtual environment is activated before running any Python scripts
5. **Dependencies**: Keep `requirements.txt` updated with all project dependencies



## Communication and Collaboration Rules
1. **Discussion First**: When user uses words like "discuss", "商量", "讨论", "考虑" etc., DO NOT write code immediately
2. **Clarification**: Always clarify requirements through dialogue before implementing complex features
3. **Confirmation**: Confirm understanding of user requirements before generating code
4. **Explanation**: Provide clear explanations of proposed solutions in Chinese (as per user's preferred language)
5. **Iterative Approach**: Break down complex tasks into smaller, manageable steps and discuss each step

## Response Guidelines
1. **User's Preferred Language**: All explanations and discussions should be in Chinese (中文)
2. **Code Comments**: Keep code comments in English as per coding standards
3. **Error Messages**: Provide error explanations in Chinese for better understanding
4. **Documentation**: Technical documentation can be in English, but explanations to user should be in Chinese
