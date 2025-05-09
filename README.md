# Economic Impact Analysis of Content-Oblivious Ordering in DeFi Liquidations

## Agent-Based Simulation of MakerDAO Liquidations with Order Fairness Protocols

This project presents an agent-based simulation built using the Mesa framework to analyze the economic impact of different content-oblivious transaction ordering mechanisms—Threshold Encryption (TE) and Verifiable Delay Functions (VDFs)—on the MakerDAO Liquidation 2.0 Dutch auction system. The simulation specifically investigates how these Order Fairness Protocols (OFPs) can mitigate Maximal Extractable Value (MEV), primarily front-running attacks targeting liquidation auction bids.

## Project Context

This simulation was developed as a project checkpoint report focusing on the economic impact analysis of content-oblivious ordering in DeFi liquidations. It builds upon literature review and a detailed research plan.

## Model Overview

The simulation models a simplified MakerDAO ecosystem including:

* **Oracle:** Provides asset price feeds.
* **MakerState:** Manages Vaults, Clipper auction contracts, and agent DAI balances.
* **Mempool:** A simulated transaction pool.
* **BlockProducer:** Processes transactions from the Mempool.
* **Agents:**
    * **Borrowers:** Users with collateralized debt positions (Vaults).
    * **Keepers:** External actors who trigger liquidations and bid in auctions.
    * **MEV Searchers:** Actors attempting to front-run profitable 'take' bids in the Baseline scenario.

The simulation steps through time, modeling price updates, agent actions (monitoring, bidding), transaction propagation via the Mempool, and block processing.

## Scenarios

The simulation compares three distinct scenarios:

1.  **Baseline Model:** Simulates a standard MakerDAO-style auction environment with gas-price prioritized transaction ordering, allowing for MEV front-running.
2.  **TE-OFP Model:** Extends the baseline by simulating encrypted transactions and introducing a post-ordering decryption delay. MEV searchers cannot read encrypted content.
3.  **VDF-OFP Model:** Extends the baseline by associating transactions with a Verifiable Delay Function, enforcing a pre-execution computational delay. Transaction content is inaccessible until the delay elapses.

## Evaluation Metrics

The simulation collects data to compare the scenarios based on metrics related to Efficiency, Fairness, and Costs:

* **Efficiency:** Auction Completion Time, Auction Price Efficiency (vs. Oracle), Total Value Liquidated, Protocol Solvency (Bad Debt).
* **Fairness:** MEV Extracted (Front-running), Borrower Loss Severity, Bidder Profit Distribution.
* **Costs:** Average End-to-End Transaction Latency, Estimated Computational Overhead (proxy).

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
