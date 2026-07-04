# Monad Blitz Sprint Plan

## Goal

Make Flying Police feel like an autonomous security agent with verifiable on-chain
memory on Monad, not just a video-analysis dashboard.

## Winning Demo Sentence

Flying Police detects a real-world security event, turns the alert into tamper-proof
evidence, anchors the proof on Monad, and lets anyone verify the agent's action.

## Must Finish Today

1. Deploy `EvidenceRegistry` on Monad Testnet.
2. Put the deployed address in `.env` as `EVIDENCE_REGISTRY_ADDRESS`.
3. Process one demo video that reliably triggers a high-severity alert.
4. Show at least one real transaction link in the UI's Monad Evidence Receipts.
5. Prepare a two-minute pitch around agent trust, evidence, and on-chain memory.

## Team Split

### Person A: Monad Proof

- Add a funded testnet wallet private key to `.env`.
- Run:

```sh
HOME="$PWD/.hardhat-home" HARDHAT_DISABLE_TELEMETRY_PROMPT=true npm run deploy:evidence
```

- Copy the printed contract address into `.env`:

```env
EVIDENCE_REGISTRY_ADDRESS=0x...
```

- Run the app and confirm receipt status changes from `NOT_CONFIGURED` to
`ANCHORED`.

### Person B: Demo Flow

- Pick the most reliable video.
- Confirm it triggers a clear rule alert.
- Keep the UI open on:
  - processed frame,
  - alert feed,
  - Monad Evidence Receipts,
  - transaction link.
- Prepare fallback screenshots in case venue Wi-Fi is rough.

## Pitch Structure

1. Problem: autonomous agents can act, but their actions are hard to trust.
2. Demo: Flying Police watches drone/security footage and detects a threat.
3. Proof: the alert becomes canonical evidence, then gets anchored on Monad.
4. Verification: the UI exposes the evidence hash and transaction link.
5. Why Monad: high throughput and low fees make frequent agent evidence practical.
6. Future: agent reputation, insurance claims, facility audits, and multi-agent
   incident response.

## Code Freeze Checklist

- Contract deployed.
- `.env` has `EVIDENCE_REGISTRY_ADDRESS`.
- UI shows at least one on-chain receipt.
- Demo video is local and tested.
- Pitch is under two minutes.
- Backup screenshots are ready.
