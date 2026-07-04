require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();
require("dotenv").config({ path: ".env.local", override: true });

const MONAD_RPC_URL = process.env.MONAD_RPC_URL || "https://testnet-rpc.monad.xyz";
const MONAD_CHAIN_ID = Number(process.env.MONAD_CHAIN_ID || 10143);
const MONAD_PRIVATE_KEY = process.env.MONAD_PRIVATE_KEY || "";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: {
      evmVersion: "prague",
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    monadTestnet: {
      url: MONAD_RPC_URL,
      chainId: MONAD_CHAIN_ID,
      accounts: MONAD_PRIVATE_KEY ? [MONAD_PRIVATE_KEY] : [],
    },
  },
};
