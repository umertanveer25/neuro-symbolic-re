# Contributing to Neuro-Symbolic Requirements Classification

We welcome contributions from researchers and practitioners in Requirements Engineering, Natural Language Processing, and Neuro-Symbolic Artificial Intelligence.

## Code of Conduct
We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful and constructive in discussions and reviews.

## Development Workflow

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Clone and Create a Branch:**
   ```bash
   git clone https://github.com/your-username/neuro-symbolic-requirements-classification.git
   cd neuro-symbolic-requirements-classification
   git checkout -b feature/your-feature-name
   ```
3. **Set Up the Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Make Code Changes & Add Tests:**
   - Any new symbolic rule or quality constraint must be accompanied by unit tests in `tests/test_symbolic.py`.
   - Any new generative pattern must enforce IEEE 830 modal operators and ISO/IEC 25010 characteristics.
5. **Run the Test Suite:**
   ```bash
   python -m unittest discover tests
   ```
6. **Submit a Pull Request:** Open a PR against `master` with a clear description of the enhancement, bug fix, or benchmark addition.
