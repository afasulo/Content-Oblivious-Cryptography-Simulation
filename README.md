# Economic Impact Analysis of Content-Oblivious Ordering in DeFi Liquidations

## Agent-Based Simulation of MakerDAO Liquidations with Order Fairness Protocols

This project presents an agent-based simulation built using the Mesa framework to analyze the economic impact of different content-oblivious transaction ordering mechanisms—Threshold Encryption (TE) and Verifiable Delay Functions (VDFs)—on the MakerDAO Liquidation 2.0 Dutch auction system[cite: 51, 54]. The simulation specifically investigates how these Order Fairness Protocols (OFPs) can mitigate Maximal Extractable Value (MEV), primarily front-running attacks targeting liquidation auction bids[cite: 51, 119, 129].

## Project Context

This simulation was developed as a project checkpoint report focusing on the economic impact analysis of content-oblivious ordering in DeFi liquidations[cite: 1]. It builds upon literature review and a detailed research plan[cite: 51].

## Model Overview

The simulation models a simplified MakerDAO ecosystem including:

* **Oracle:** Provides asset price feeds[cite: 366].
* **MakerState:** Manages Vaults, Clipper auction contracts, and agent DAI balances[cite: 367, 368, 369, 370].
* **Mempool:** A simulated transaction pool[cite: 371].
* **BlockProducer:** Processes transactions from the Mempool[cite: 373].
* **Agents:**
    * **Borrowers:** Users with collateralized debt positions (Vaults)[cite: 364].
    * **Keepers:** External actors who trigger liquidations and bid in auctions[cite: 364, 106].
    * **MEV Searchers:** Actors attempting to front-run profitable 'take' bids in the Baseline scenario[cite: 364, 365, 122].

The simulation steps through time, modeling price updates, agent actions (monitoring, bidding), transaction propagation via the Mempool, and block processing.

## Scenarios

The simulation compares three distinct scenarios[cite: 18]:

1.  **Baseline Model:** Simulates a standard MakerDAO-style auction environment with gas-price prioritized transaction ordering, allowing for MEV front-running[cite: 18, 19].
2.  **TE-OFP Model:** Extends the baseline by simulating encrypted transactions and introducing a post-ordering decryption delay[cite: 20]. MEV searchers cannot read encrypted content[cite: 21].
3.  **VDF-OFP Model:** Extends the baseline by associating transactions with a Verifiable Delay Function, enforcing a pre-execution computational delay[cite: 21, 22]. Transaction content is inaccessible until the delay elapses[cite: 22].

## Evaluation Metrics

The simulation collects data to compare the scenarios based on metrics related to Efficiency, Fairness, and Costs[cite: 23, 306]:

* **Efficiency:** Auction Completion Time, Auction Price Efficiency (vs. Oracle), Total Value Liquidated, Protocol Solvency (Bad Debt)[cite: 23, 307].
* **Fairness:** MEV Extracted (Front-running), Borrower Loss Severity, Bidder Profit Distribution[cite: 24, 321].
* **Costs:** Average End-to-End Transaction Latency, Estimated Computational Overhead (proxy)[cite: 25, 334].

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.7+
* `pip` (Python package installer)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/](https://github.com/YourUsername/)[Choose Your Repo Name Here].git
    cd [Choose Your Repo Name Here]
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the required packages:
    ```bash
    pip install mesa numpy pandas
    ```

### Running the Simulation

Execute the main simulation script from the project directory with your virtual environment activated:

```bash
python simulation.py
