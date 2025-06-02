import json
from web3 import Web3
from pathlib import Path

# === CONFIGURATION ===
RPC_URL = "https://base-mainnet.infura.io/v3/d2f72550e4bc42168e4ecdbbe8f1bd95"
# VAULT_ADDRESS = "0x1487d907247e6e1bCfb6C73B193c74a16266368C"
# POOL_ADDRESS = "0xaBfF72aEE1bA72fc459acd5222dd84a3182411BB"

# === SETUP WEB3 ===
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# === LOAD ABI ===


def load_abi(name: str):
    path = Path("abi") / f"{name}.json"
    with open(path) as f:
        return json.load(f)['abi']
    

def get_contracts(vault_address):
    vault_abi = load_abi("AlgebraVault")
    vault = web3.eth.contract(address=Web3.to_checksum_address(vault_address), abi=vault_abi)

    pool_abi = load_abi("AlgebraPool")
    pool_address = vault.functions.pool().call()
    pool = web3.eth.contract(address=pool_address, abi=pool_abi)
    
    return vault, pool


def get_positions_amounts(vault):    
    base_liquidity, base_amount0, base_amount1 = vault.functions.getBasePosition().call()
    limit_liquidity, limit_amount0, limit_amount1 = vault.functions.getLimitPosition().call()
    return base_amount0, limit_amount0, base_amount1, limit_amount1, base_liquidity, limit_liquidity


def get_pool_price(pool):
    price, tick, *_ = pool.functions.globalState().call()
    return (price * price) / (2 ** 192), tick

def get_pos_ticks(vault):
    base_lower = vault.functions.baseLower().call()
    base_upper = vault.functions.baseUpper().call()
    limit_lower = vault.functions.limitLower().call()
    limit_upper = vault.functions.limitUpper().call()
    
    return base_lower, base_upper, limit_lower, limit_upper

def get_vault_tokens(vault):
    token0 = vault.functions.token0().call()
    token1 = vault.functions.token1().call()
    allowToken0 = vault.functions.allowToken0().call()
    
    return token0, token1, allowToken0

def get_token_name(token_address):
    token_abi = load_abi("ERC20")
    token = web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=token_abi)
    
    return token.functions.symbol().call()
