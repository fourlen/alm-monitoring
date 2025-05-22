import json
from web3 import Web3
from pathlib import Path

# === CONFIGURATION ===
RPC_URL = "https://base-mainnet.infura.io/v3/d2f72550e4bc42168e4ecdbbe8f1bd95"
VAULT_ADDRESS = "0x1487d907247e6e1bCfb6C73B193c74a16266368C"
POOL_ADDRESS = "0xaBfF72aEE1bA72fc459acd5222dd84a3182411BB"

# === SETUP WEB3 ===
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# === LOAD ABI ===


def load_abi(name: str):
    path = Path("abi") / f"{name}.json"
    with open(path) as f:
        return json.load(f)['abi']


vault_abi = load_abi("AlgebraVault")
pool_abi = load_abi("AlgebraPool")
vault = web3.eth.contract(address=VAULT_ADDRESS, abi=vault_abi)
pool = web3.eth.contract(address=POOL_ADDRESS, abi=pool_abi)


def get_positions_amounts():
    base_liquidity, base_amount0, base_amount1 = vault.functions.getBasePosition().call()
    limit_liquidity, limit_amount0, limit_amount1 = vault.functions.getLimitPosition().call()
    return base_amount0 + limit_amount0, base_amount1 + limit_amount1


def get_pool_price():
	price, tick, *_ = pool.functions.globalState().call()
	return (price * price) / (2 ** 192)
