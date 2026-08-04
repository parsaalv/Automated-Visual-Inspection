"""Centralized logging configuration for the Digital Twin simulation.

Importing this module configures the root logging format exactly once and
exposes a shared ``logger`` instance so every other module in the project
logs consistently under the same ``"DigitalTwin"`` namespace.
"""

import logging

# --- Logging setup (used to log camera shots and vision inference results) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DigitalTwin")
