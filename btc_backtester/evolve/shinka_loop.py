import os
import json
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ShinkaOrchestrator:
    def __init__(self, ledger_path="ledger.json"):
        self.ledger_path = ledger_path
        self.ledger = self._load_ledger()
        self.acceptance_rules = {
            "min_sharpe_delta": +0.05,
            "max_dd_regression": -2.0,
            "min_trades": 200,
        }

    def _load_ledger(self) -> Dict[str, Any]:
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r") as f:
                return json.load(f)
        return {"champion": None, "history": []}

    def save_ledger(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.ledger, f, indent=2)

    def build_context_packet(self, champion: Dict[str, Any], version: int) -> str:
        # Template from §4.2
        # We'll need to read some files for this
        strategy_code = ""
        try:
            with open("btc_backtester/models/xgb_strategy.py", "r") as f:
                strategy_code = f.read()
        except: pass

        return f"""
You are a quantitative researcher improving a BTC futures trading strategy.

## Current Champion (v{champion.get('strategy_version', 'v0')})
- Sharpe (annual): {champion.get('sharpe_annual', 0)}
- Calmar: {champion.get('calmar_ratio', 0)}
- Max DD: {champion.get('max_drawdown_pct', 0)}%
- Win rate: {champion.get('win_rate', 0)}%
- Feature count: {len(champion.get('feature_set', []))}

## Strategy Code Snapshot
```python
{strategy_code[:2000]} # Truncated for context
```

## Task
Propose exactly ONE mutation as a unified git diff / Python patch.
Respond with <MUTATION_TYPE>, <HYPOTHESIS>, and <PATCH> tags.
"""

    def apply_patch(self, patch: str):
        """Applies a patch. If it's a python statement, we might need a different approach."""
        if patch.startswith("diff") or "---" in patch:
            # Use git apply
            with open("temp.patch", "w") as f:
                f.write(patch)
            subprocess.run(["git", "apply", "temp.patch"], check=True)
            os.remove("temp.patch")
        else:
            # Assume it's a code modification we should apply manually or via some logic
            # For simplicity in this prototype, let's say we only handle simple replacements
            logger.warning("Non-diff patch received. Manual application logic needed.")

    def revert_patch(self):
        subprocess.run(["git", "checkout", "."], check=True)

    def accept(self, champion: Dict[str, Any], challenger: Dict[str, Any]) -> bool:
        if champion is None: return True
        
        sharpe_ok = challenger["sharpe_annual"] >= champion["sharpe_annual"] + self.acceptance_rules["min_sharpe_delta"]
        dd_ok = challenger["max_drawdown_pct"] >= champion["max_drawdown_pct"] + self.acceptance_rules["max_dd_regression"]
        trades_ok = challenger["trades_total"] >= self.acceptance_rules["min_trades"]
        
        return sharpe_ok and dd_ok and trades_ok

    def update_champion(self, challenger: Dict[str, Any], mutation: Dict[str, Any]):
        self.ledger["champion"] = challenger
        self.ledger["history"].append({
            "metrics": challenger,
            "mutation": mutation
        })
        self.save_ledger()
        
        # Git commit
        msg = f"[ShinkáEvolve {challenger['strategy_version']}] {mutation['type']}: {mutation['hypothesis']}"
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=True)
