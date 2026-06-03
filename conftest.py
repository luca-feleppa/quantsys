# IT: Configurazione globale pytest per QUANTSYS.
# EN: Global pytest configuration for QUANTSYS.
#
# IT: Esempi di esecuzione:
# EN: Run examples:
#   pytest tests/ -v                    # IT: verbose | EN: verbose
#   pytest tests/ -v --timeout=60       # IT: timeout per test HMM | EN: timeout for HMM tests
#   pytest tests/ -k "not TestHMM"      # IT: salta test hmmlearn | EN: skip hmmlearn tests
#   pytest tests/ -x                    # IT: stop al primo errore | EN: stop on first failure

import sys
from pathlib import Path

# IT: Aggiunge la root al sys.path: import "quantsys.xxx" senza pip install -e .
# EN: Add project root to sys.path so "quantsys.xxx" imports work without pip install -e .
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
