# Contributing Guidelines

Thank you for your interest in contributing to the Rural Emergency Triage AI Assistant!

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include system info, error messages, and steps to reproduce
- Check existing issues before creating new ones

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code (`black src/` and `isort src/`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions focused and small

### Testing
- Write unit tests for new features
- Maintain >80% code coverage
- Test edge cases

## Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rural-emergency-triage-ai.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Questions?

Open an issue or contact the maintainers.
