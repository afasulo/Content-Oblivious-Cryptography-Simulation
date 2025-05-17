# simulation_project/core/clipper.py

from .auction import Auction
from .. import config # Relative import for config.VERBOSE_LOGGING

class Clipper:
    """
    Simulates the Clipper contract for one collateral type (ilk).
    Manages the lifecycle of Dutch auctions (kick, take, check_stale).
    """
    def __init__(self, ilk: str, model, pvt_params: dict, buf: float,
                 tail: int, cusp: float, tip: float, chip: float):
        self.ilk = ilk
        self.model = model
        self.active_auctions = {}
        self.next_auction_id_counter = 0

        self.pvt_curve_params = pvt_params
        self.buf = float(buf)
        self.tail = int(tail)
        self.cusp_factor = float(cusp)
        self.tip = float(tip)
        self.chip = float(chip)

    def _generate_auction_id(self) -> str:
        auction_id = f"{self.ilk}-auction-{self.next_auction_id_counter}"
        self.next_auction_id_counter += 1
        return auction_id

    def kick(self, vault_urn: str, tab_debt: float, lot_collateral: float, current_time: float) -> tuple:
        auction_id = self._generate_auction_id()
        oracle_price = self.model.oracle.current_price

        if oracle_price <= 0:
            # This is a critical failure condition for kicking, so it should probably always be logged.
            print(f"    !!! [Time {current_time:.0f}] Clipper ({self.ilk}): Cannot kick auction {auction_id}. Oracle price <= 0 ({oracle_price:.2f}) !!!")
            return None, 0, 0

        start_price = oracle_price * self.buf

        new_auction = Auction(
            auction_id=auction_id,
            ilk=self.ilk,
            lot_collateral=lot_collateral,
            tab_debt=tab_debt,
            start_time=current_time,
            start_price=start_price,
            pvt_curve_params=self.pvt_curve_params,
            duration_limit=self.tail,
            reset_threshold_factor=self.cusp_factor,
            model=self.model
        )
        self.active_auctions[auction_id] = new_auction

        if config.VERBOSE_LOGGING:
            print(f"    >>> [Time {current_time:.0f}] Clipper ({self.ilk}): KICKED Auction {auction_id}. "
                  f"Lot: {lot_collateral:.4f}, Tab: {tab_debt:.2f}, Start Price: {start_price:.2f} <<<")

        self.model.total_value_liquidated += tab_debt
        return auction_id, self.tip, self.chip

    def take(self, auction_id: str, bidder_id: str, amt_collateral_to_buy: float,
             max_price_per_unit: float, bidder_dai_balance: float, current_time: float) -> tuple:
        if auction_id not in self.active_auctions:
            return False, 0, 0, "Auction not found"

        auction = self.active_auctions[auction_id]
        if auction.status != "Active":
            return False, 0, 0, f"Auction status is {auction.status}, not Active"

        current_auction_price = auction.get_current_price(current_time)

        if current_auction_price <= 1e-9:
            return False, 0, 0, "Auction price is zero or negligible"

        if current_auction_price > max_price_per_unit:
            if config.VERBOSE_LOGGING and self.model.ofp_mode in ['TE', 'VDF']:
                 print(f"      (Info {self.model.ofp_mode}): Bid failed for {bidder_id} on {auction_id} - "
                       f"Auction Price {current_auction_price:.2f} > Max Bid Price {max_price_per_unit:.2f} (Latency effect?)")
            return False, 0, 0, "Price too high (auction price > max bid price)"

        actual_amt_collateral = min(amt_collateral_to_buy, auction.remaining_lot)
        dai_needed_for_remaining_tab = max(0, auction.tab - auction.dai_raised)
        max_collateral_for_tab = dai_needed_for_remaining_tab / current_auction_price if current_auction_price > 1e-9 else float('inf')
        actual_amt_collateral = min(actual_amt_collateral, max_collateral_for_tab)
        dai_cost_for_this_bid = actual_amt_collateral * current_auction_price

        if dai_cost_for_this_bid > bidder_dai_balance:
            affordable_collateral = bidder_dai_balance / current_auction_price if current_auction_price > 1e-9 else 0
            actual_amt_collateral = min(actual_amt_collateral, affordable_collateral)
            dai_cost_for_this_bid = actual_amt_collateral * current_auction_price

        if actual_amt_collateral <= 1e-9:
            return False, 0, 0, "Insufficient balance or adjusted bid amount too small"

        auction.remaining_lot -= actual_amt_collateral
        auction.dai_raised += dai_cost_for_this_bid
        auction.bids.append({
            'bidder': bidder_id,
            'time': current_time,
            'amount_collateral': actual_amt_collateral,
            'price_paid': current_auction_price,
            'dai_cost': dai_cost_for_this_bid
        })

        if config.VERBOSE_LOGGING:
            print(f"    >>> [Time {current_time:.0f}] Bidder {bidder_id} SUCCESSFUL take on {auction_id}. "
                  f"Bought: {actual_amt_collateral:.4f} ETH @ {current_auction_price:.2f} DAI/ETH. "
                  f"Cost: {dai_cost_for_this_bid:.2f}. Raised: {auction.dai_raised:.2f}/{auction.tab:.2f} <<<")

        if auction.dai_raised >= auction.tab - 1e-6:
            auction.status = "Completed"
            if config.VERBOSE_LOGGING:
                print(f"    >>> [Time {current_time:.0f}] Auction {auction_id} COMPLETED. "
                      f"Raised: {auction.dai_raised:.2f}/{auction.tab:.2f}. "
                      f"Collateral Left: {auction.remaining_lot:.4f} <<<")
            self.model.record_auction_end(auction)

        return True, actual_amt_collateral, dai_cost_for_this_bid, "Success"

    def check_stale_auctions(self, current_time: float):
        for auction_id, auction in list(self.active_auctions.items()):
            if auction.status != "Active":
                continue

            elapsed_time = auction.get_elapsed_time(current_time)
            current_auction_price = auction.get_current_price(current_time)

            reset_price_threshold = auction.start_price * auction.reset_threshold_factor
            is_stale_by_cusp = (current_auction_price > 0 and
                                current_auction_price < reset_price_threshold and
                                elapsed_time > 0)

            is_stale_by_tail = elapsed_time > auction.duration_limit

            if is_stale_by_tail or is_stale_by_cusp:
                reason = "DURATION (tail)" if is_stale_by_tail else "PRICE (cusp)"
                # Stale auction print is important for debugging auction mechanics, keep unconditional for now
                print(f"    !!! [Time {current_time:.0f}] Auction {auction_id} hit {reason} limit. "
                      f"Status: Failed. Raised: {auction.dai_raised:.2f}/{auction.tab:.2f} !!!")
                auction.status = "Failed"
                self.model.record_auction_end(auction)
