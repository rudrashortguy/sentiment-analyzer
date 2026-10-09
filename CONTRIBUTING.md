# Contributing to Sentiment Analyzer

Thank you for your interest in contributing!

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests:
   ```bash
   pytest tests/
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and small

## Commit Messages

Follow Conventional Commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `test:` - Test changes
- `docs:` - Documentation changes

## Pull Requests

- Create a feature branch
- Ensure all tests pass
- Update README if adding new features
- Keep changes minimal and focused