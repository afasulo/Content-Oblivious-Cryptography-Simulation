# simulation_project/core/market_state.py

from .clipper import Clipper
from .vault import Vault
from .. import config # Relative import

class MakerState:
    """
    Holds the overall state of the simulated MakerDAO system, including
    all vaults, clipper contracts (per ilk), and agent DAI balances.
    """
    def __init__(self, model):
        self.model = model # Mesa model instance
        self.vaults = {} # Stores Vault objects, keyed by vault_owner_id
        self.clippers = {} # Stores Clipper objects, keyed by ilk string (e.g., 'ETH-A')
        self.agent_dai_balances = {} # Stores DAI balances of agents, keyed by agent_id

        self._initialize_clippers()

    def _initialize_clippers(self):
        """Initializes default clipper(s) based on config."""
        # Example for 'ETH-A' ilk
        eth_a_pvt_params = {
            'type': config.PVT_CURVE_TYPE,
            'step': config.PVT_STEP_DURATION,
            'cut': config.PVT_STEP_DECAY,
            'tau': config.PVT_LINEAR_TAU
        }
        self.clippers['ETH-A'] = Clipper(
            ilk='ETH-A', 
            model=self.model,
            pvt_params=eth_a_pvt_params,
            buf=config.PRICE_BUFFER, 
            tail=config.AUCTION_DURATION_LIMIT, 
            cusp=config.AUCTION_RESET_THRESHOLD,
            tip=config.KEEPER_INCENTIVE_TIP, 
            chip=config.KEEPER_INCENTIVE_CHIP
        )
        # Add other clippers for different ilks if needed

    def add_vault(self, vault: Vault):
        """Adds a new vault to the system state."""
        if vault.owner_id in self.vaults:
            print(f"Warning: Vault with owner_id {vault.owner_id} already exists. Overwriting.")
        self.vaults[vault.owner_id] = vault

    def get_vault(self, owner_id: str) -> Vault | None:
        """Retrieves a vault by its owner's ID."""
        return self.vaults.get(owner_id)

    def add_agent_balance(self, agent_id, amount: float):
        """Initializes or sets an agent's DAI balance."""
        self.agent_dai_balances[agent_id] = float(amount)

    def get_agent_balance(self, agent_id) -> float:
        """Gets an agent's DAI balance. Returns 0 if agent not found."""
        return self.agent_dai_balances.get(agent_id, 0.0)

    def update_agent_balance(self, agent_id, change: float):
        """
        Updates an agent's DAI balance by a given change (can be positive or negative).
        Ensures balance does not go below zero.
        """
        current_balance = self.get_agent_balance(agent_id)
        self.agent_dai_balances[agent_id] = max(0, current_balance + float(change))

    def get_clipper(self, ilk: str) -> Clipper | None:
        """Retrieves a clipper contract by its ilk."""
        return self.clippers.get(ilk)

    def get_all_active_auctions(self) -> list:
        """Returns a list of all active auctions across all clippers."""
        all_auctions = []
        for clipper in self.clippers.values():
            all_auctions.extend(clipper.active_auctions.values())
        return [auc for auc in all_auctions if auc.status == "Active"]

