# Monad Integration Setup

Flying Police can run fully offline. When Monad settings are present, every rule alert
is converted into a canonical evidence payload, hashed, and anchored through
`EvidenceRegistry`.

## 1. Deploy The Registry

### Option A: Deploy From Terminal Without MetaMask

This is the recommended hackathon path if MetaMask refuses to add Monad Testnet.

Install the contract tooling:

```sh
npm install
```

Create or update `.env`:

```env
MONAD_RPC_URL=https://testnet-rpc.monad.xyz
MONAD_CHAIN_ID=10143
MONAD_PRIVATE_KEY=<private key for the funded test wallet>
MONAD_EXPLORER_TX_URL=https://testnet.monadscan.com/tx
```

If the first RPC is unavailable, use:

```env
MONAD_RPC_URL=https://rpc.ankr.com/monad_testnet
```

Compile and deploy:

```sh
npm run compile:contracts
npm run deploy:evidence
```

If Hardhat cannot create its local preference folder in this environment, run the
same commands with a workspace-local home:

```sh
HOME="$PWD/.hardhat-home" HARDHAT_DISABLE_TELEMETRY_PROMPT=true npm run compile:contracts
HOME="$PWD/.hardhat-home" HARDHAT_DISABLE_TELEMETRY_PROMPT=true npm run deploy:evidence
```

The deploy script prints:

```text
Contract address: 0x...
Transaction hash: 0x...
```

Copy the contract address into `.env`:

```env
EVIDENCE_REGISTRY_ADDRESS=0x...
```

### Option B: Deploy With Remix

Deploy `contracts/src/EvidenceRegistry.sol` to Monad using your preferred EVM
tooling. Monad supports standard Ethereum-style contracts and tooling, so a
simple Foundry, Remix, Hardhat, or third-party deploy flow works.

Keep the deployed contract address.

## 2. Configure `.env`

Copy `.env.example` to `.env` if needed and fill:

```env
MONAD_RPC_URL=<your Monad RPC URL>
MONAD_CHAIN_ID=<Monad network chain id>
MONAD_PRIVATE_KEY=<operator wallet private key>
EVIDENCE_REGISTRY_ADDRESS=<deployed EvidenceRegistry address>
MONAD_EXPLORER_TX_URL=<explorer tx URL prefix, optional>
```

Leave these blank during local development. The app will still generate evidence
hashes and record `not_configured` anchor attempts in SQLite.

## 3. Run Flying Police

```sh
streamlit run ui/app.py
```

Process a video that triggers at least one rule alert. The session log will show
a line like:

```text
Monad evidence: anchored 0x123456789abc... tx=0xabcdef123456...
```

If the Monad values are blank, it will show:

```text
Monad evidence: not_configured 0x123456789abc...
```

## 4. Demo Story

1. Flying Police detects suspicious motion.
2. Deterministic rules fire before any LLM reasoning.
3. The alert becomes canonical JSON.
4. The JSON hash is anchored on Monad.
5. Anyone with the original evidence can recompute the hash and verify that it
   matches the on-chain record.
