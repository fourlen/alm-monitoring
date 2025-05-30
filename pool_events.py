from gql import gql

def get_pool_swaps(client, pool):
	query = gql("""
    query Swaps($pool: String!) {
		swaps(where: { timestamp_gte: "0", pool: $pool }) {
			id
			timestamp
			price
			pool {
				id
			}
		}
	}
    """)
	return client.execute(query, variable_values={'pool': pool})['swaps']
