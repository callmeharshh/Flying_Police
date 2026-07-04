const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  if (!deployer) {
    throw new Error(
      "No deployer account found. Set MONAD_PRIVATE_KEY in .env before deploying."
    );
  }

  const network = await hre.ethers.provider.getNetwork();
  console.log(`Network: ${network.name} (${network.chainId})`);
  console.log(`Deployer: ${deployer.address}`);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log(`Balance: ${hre.ethers.formatEther(balance)} MON`);
  if (balance === 0n) {
    throw new Error("Deployer has 0 MON. Fund this address from the Monad faucet first.");
  }

  const EvidenceRegistry = await hre.ethers.getContractFactory("EvidenceRegistry");
  const registry = await EvidenceRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  const tx = registry.deploymentTransaction();

  console.log("");
  console.log("EvidenceRegistry deployed");
  console.log(`Contract address: ${address}`);
  console.log(`Transaction hash: ${tx.hash}`);
  console.log("");
  console.log("Add this to .env:");
  console.log(`EVIDENCE_REGISTRY_ADDRESS=${address}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
