# Pull Request Template

## Description

<!-- Please provide a clear and concise description of the changes you are proposing -->

### Type of Change

Please select the relevant option(s):

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📖 Documentation update (changes to documentation)
- [ ] 🎨 Code style/formatting (non-functional changes)
- [ ] ♻️ Code refactoring (no functional changes, no api changes)
- [ ] ⚡ Performance improvements
- [ ] 🧪 Test additions or improvements
- [ ] 🔧 Build/CI configuration changes

## Related Issues

<!-- Link to related issues using: Closes #issue_number, Fixes #issue_number, or Resolves #issue_number -->

## Changes Made

<!-- Please provide a detailed description of the changes -->

### Core Changes
- [ ] Modified existing functionality
- [ ] Added new classes/modules
- [ ] Updated algorithms or processing methods
- [ ] Changed API or public interfaces

### Files Modified
<!-- List the main files that were changed and why -->

## Code Quality Checklist

### Testing Requirements ✅
- [ ] All existing tests pass
- [ ] New functionality includes appropriate unit tests
- [ ] Tests follow Test-Driven Design (TDD) principles when applicable
- [ ] Test coverage maintained or improved
- [ ] Manual testing performed (if applicable)

### Code Standards 📋
- [ ] Code follows the project's coding standards
- [ ] Functions and classes have Google-style docstrings
- [ ] Code has been formatted with `blue` formatter
- [ ] Imports are sorted with `isort`
- [ ] Code passes linting checks (`task lint-check`)
- [ ] No unnecessary dependencies added

### Documentation 📚
- [ ] Code includes appropriate inline documentation
- [ ] Public APIs are properly documented
- [ ] Examples provided in docstrings when applicable
- [ ] Documentation updated for new features (if applicable)

## Branch Strategy

### Target Branch
- [ ] Targeting `develop` branch (for new features/non-critical fixes)
- [ ] Targeting `main` branch (for critical hotfixes only)

**Note:** According to our [contribution guidelines](docs/contribute.md), please target the `develop` branch for most contributions to ensure proper testing before merging to `main`.

## Breaking Changes Assessment

- [ ] This PR introduces no breaking changes
- [ ] This PR introduces breaking changes (please detail below)

<!-- If introducing breaking changes, please describe:
- What breaks
- Migration path for existing users
- Version bump implications (major.minor.patch)
-->

## Testing Instructions

<!-- Provide clear instructions on how reviewers can test your changes -->

### Environment Setup
```bash
poetry shell && poetry install
task test
```

### Specific Testing Steps
<!-- List specific steps to test the new functionality -->

1. 
2. 
3. 

## Performance Impact

- [ ] No performance impact
- [ ] Performance improvements
- [ ] Potential performance regression (please justify)

## Additional Context

<!-- Add any other context about the pull request here -->

### Dependencies
<!-- List any new dependencies and justify their inclusion -->

### Compatibility
<!-- Note any compatibility considerations -->

---

## Checklist for Reviewers

- [ ] Code review completed
- [ ] Tests pass locally
- [ ] Documentation is clear and complete
- [ ] Change aligns with project goals
- [ ] Appropriate branch target confirmed

---

**By submitting this pull request, I confirm:**
- [ ] I have read and followed the [contribution guidelines](docs/contribute.md)
- [ ] I have tested my changes thoroughly
- [ ] I understand this is an open source project and my contributions may be used by others
- [ ] I agree to the project's [Code of Conduct](CODE_OF_CONDUCT.md)

<!-- Thank you for contributing to asltk! 🚀 -->