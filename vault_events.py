from gql import gql

def get_vault_rebalances(client, vault_id):
    query = gql("""
    query ($vault: Bytes!) {
      vaultRebalances(where: {vault: $vault}, orderBy: createdAtTimestamp, orderDirection: desc) {
        createdAtTimestamp
        type: __typename
        totalAmount0
        totalAmount1
        feeAmount0
        feeAmount1
        lastPrice
      }
    }
    """)
    return client.execute(query, variable_values={"vault": vault_id})['vaultRebalances']

def get_vault_deposits(client, vault_id):
    query = gql("""
    query ($vault: Bytes!) {
      vaultDeposits(where: {vault: $vault}, orderBy: createdAtTimestamp, orderDirection: desc) {
        createdAtTimestamp
        type: __typename
        sender
        totalAmount0
        totalAmount1
        shares
		    lastPrice
      }
    }
    """)
    return client.execute(query, variable_values={"vault": vault_id})['vaultDeposits']

def get_vault_withdrawals(client, vault_id):
    query = gql("""
    query ($vault: Bytes!) {
      vaultWithdraws(where: {vault: $vault}, orderBy: createdAtTimestamp, orderDirection: desc) {
        createdAtTimestamp
        type: __typename
        sender
        totalAmount0
        totalAmount1
        shares
        lastPrice
      }
    }
    """)
    return client.execute(query, variable_values={"vault": vault_id})['vaultWithdraws']

def get_vault_collect_fees(client, vault_id):
    query = gql("""
    query ($vault: Bytes!) {
      vaultCollectFees(where: {vault: $vault}, orderBy: createdAtTimestamp, orderDirection: desc) {
        createdAtTimestamp
        type: __typename
        feeAmount0
        feeAmount1
        lastPrice
		totalAmount0
        totalAmount1
      }
    }
    """)
    return client.execute(query, variable_values={"vault": vault_id})['vaultCollectFees']

def get_all_vault_events(client, vault_id):
    rebalances = get_vault_rebalances(client, vault_id)
    deposits = get_vault_deposits(client, vault_id)
    withdrawals = get_vault_withdrawals(client, vault_id)
    collect_fees = get_vault_collect_fees(client, vault_id)

    all_events = rebalances + deposits + withdrawals + collect_fees
    all_events.sort(key=lambda x: int(x['createdAtTimestamp']))

    return all_events, rebalances
