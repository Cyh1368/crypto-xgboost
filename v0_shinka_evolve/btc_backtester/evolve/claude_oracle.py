import subprocess
import re
import json
import logging
import time

logger = logging.getLogger(__name__)

class ClaudeOracle:
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.model = model

    def query_claude(self, context: str) -> dict:
        """
        Calls Claude CLI subprocess.
        Returns parsed mutation dict: {type, hypothesis, patch}.
        """
        # Using 'claude' command if available, or fallback to a dummy/mock if not
        # In this environment, we'll try to use the CLI as requested.
        try:
            # Note: We use -p for prompt, but Claude CLI might vary.
            # Following the prompt's skeleton:
            result = subprocess.run(
                ["claude", "-p", context, "--output-format", "text"],
                capture_output=True, text=True, timeout=180, check=True
            )
            raw = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to call Claude CLI: {e}")
            # Mock return for development if CLI is missing
            return self._mock_mutation()

        mutation_type = re.search(r"<MUTATION_TYPE>(.*?)</MUTATION_TYPE>", raw, re.S)
        hypothesis    = re.search(r"<HYPOTHESIS>(.*?)</HYPOTHESIS>",       raw, re.S)
        patch         = re.search(r"<PATCH>(.*?)</PATCH>",                 raw, re.S)

        return {
            "type":       mutation_type.group(1).strip() if mutation_type else "UNKNOWN",
            "hypothesis": hypothesis.group(1).strip()    if hypothesis    else "",
            "patch":      patch.group(1).strip()         if patch         else "",
        }

    def _mock_mutation(self):
        """Mock mutation for when Claude CLI is not available."""
        return {
            "type": "MODEL_PARAM",
            "hypothesis": "Decrease learning rate to 0.02 for better convergence.",
            "patch": "XGB_PARAMS['learning_rate'] = 0.02"
        }
