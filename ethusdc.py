import datetime
from vault_events import get_all_vault_events
from pool_events import get_pool_swaps
from chaindata import get_positions_amounts, get_pool_price, get_pos_ticks, get_contracts
import streamlit as st
import pandas as pd
import plotly.express as px
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import numpy as np
import matplotlib.pyplot as plt


# Настройка клиента GraphQL
transport = RequestsHTTPTransport(
    url="https://api.studio.thegraph.com/query/50593/clamm-alm/version/latest",
    verify=True,
    retries=3,
)
client = Client(transport=transport, fetch_schema_from_transport=True)

transport_analytics = RequestsHTTPTransport(
    url="https://gateway.thegraph.com/api/4d7b59e4fd14365ae609945af85f3938/subgraphs/id/5pwQHNUqE7GFeG7C32m2HM3vhXvoQpet4HAinmHFmxW5",
    verify=True,
    retries=3,
)
client_analytics = Client(transport=transport_analytics, fetch_schema_from_transport=True)

# Функция для получения данных о волте
def get_vault_data(vault_id):
    query = gql("""
    query ($id: ID!) {
      almVault(id: $id) {
        id
        totalAmount0
        totalAmount1
        decimals0
        decimals1
        lastPrice
        holdersCount
        token0
        token1
      }
    }
    """)
    params = {"id": vault_id}
    result = client.execute(query, variable_values=params)
    return result['almVault']

def process_rebalance_data(rebalances, decimals0, decimals1):
    data = []
    for r in rebalances:
        timestamp = int(r['createdAtTimestamp'])
        date = datetime.datetime.utcfromtimestamp(timestamp)
        amount0 = int(r['totalAmount0']) / (10 ** decimals0)
        amount1 = int(r['totalAmount1']) / (10 ** decimals1)
        fee0 = int(r['feeAmount0']) / (10 ** decimals0)
        fee1 = int(r['feeAmount1']) / (10 ** decimals1)
        price = float(r['lastPrice'])
        tvl = amount0 * price + amount1
        fees = fee0 * price + fee1
        ratio = amount1 / (amount0 * price + amount1) * \
            100 if (amount0 * price + amount1) > 0 else 0
        data.append({
            "date": date,
            "TVL": tvl,
            "Fees": fees,
            "Deposit Token Ratio (%)": ratio
        })
    return pd.DataFrame(data)


def build_tvl_series(events, token0Decimals, token1Decimals):
    tvl_by_time = []

    for event in events:
        timestamp = int(event['createdAtTimestamp'])
        dt = datetime.datetime.fromtimestamp(timestamp)

        tvl_usd = int(event['totalAmount0']) * float(event['lastPrice']) / (
            10 ** token0Decimals) + int(event['totalAmount1']) / (10 ** token1Decimals)

        tvl_by_time.append((dt, tvl_usd))

    return tvl_by_time


def plot_tvl(tvl_data):
    if not tvl_data:
        st.warning("Нет данных для отображения TVL.")
        return

    # Преобразуем данные в DataFrame
    df = pd.DataFrame(tvl_data, columns=["timestamp", "TVL"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")

    # Сортируем по дате
    df = df.sort_values("date")

    # Строим график
    fig_tvl = px.line(df, x="date", y="TVL", title="TVL Over Time (USD)")
    fig_tvl.update_layout(xaxis_title="Date",
                          yaxis_title="TVL (USD)", height=500)

    # Показываем в Streamlit
    st.plotly_chart(fig_tvl, use_container_width=True)


def process_update_data(updates, decimals0, decimals1):
    data = []
    for u in updates:
        timestamp = int(u['createdAtTimestamp'])
        date = datetime.datetime.utcfromtimestamp(timestamp)
        amount0 = int(u['totalAmount0']) / (10 ** decimals0)
        amount1 = int(u['totalAmount1']) / (10 ** decimals1)
        fee0 = int(u['feeAmount0']) / (10 ** decimals0) if 'feeAmount0' in u else 0
        fee1 = int(u['feeAmount1']) / (10 ** decimals1) if 'feeAmount0' in u else 0
        price = float(u['lastPrice'])
        tvl = amount0 * price + amount1
        fees = fee0 * price + fee1
        ratio = amount1 / (amount0 * price + amount1) * \
            100 if (amount0 * price + amount1) > 0 else 0
        data.append({
            "date": date,
            "TVL": tvl,
            "Fees": fees,
            "Deposit Token Ratio (%)": ratio
        })
    return pd.DataFrame(data)

def process_swaps(swaps):
    data = []
    
    if len(swaps) <= 2:
        return data

    for i in range(len(swaps)):
        timestamp = int(swaps[i]['timestamp'])
        date = datetime.datetime.utcfromtimestamp(timestamp)
        price1 = int(swaps[i - 1]['price']) ** 2 / (2 ** 192)
        price2 = int(swaps[i]['price']) ** 2 / (2 ** 192)
        data.append({
            "date": date,
            "Price change (%)": (price2 - price1) / price1
        })
    
    return data

def transform_liquidity(x):
    """ Apply a piecewise transformation to compress large negative values. """
    return np.sign(x) * np.log10(abs(x) + 1) * 20000  # Logarithmic compression

# Function to transform tick values
def transform_ticks(token0range, token1range):
    if token0range[0] == -887220:
        return [token0range[1] - (token1range[1] - token1range[0]), token0range[1]], token1range
    elif token0range[1] == 887220:
        return [token0range[0], token0range[0] + (token1range[1] - token1range[0])], token1range
    elif token1range[0] == -887220:
        return token0range, [token1range[1] - (token0range[1] - token0range[0]), token1range[1]]
    elif token1range[1] == 887220:
        return token0range, [token1range[0], token1range[0] + (token0range[1] - token0range[0])]
    else:
        return token0range, token1range

def plot_positions(token0_pos, token1_pos, decimals0, decimals1):
    tick_values = set()
    tick_positions = set()

    fig, ax = plt.subplots()

    transformed_ticks0, transformed_ticks1 = None, None
    if token0_pos and token1_pos:
        transformed_ticks0, transformed_ticks1 = transform_ticks(
            [token0_pos["bottomTick"], token0_pos["topTick"]],
            [token1_pos["bottomTick"], token1_pos["topTick"]]
        )
    elif token0_pos:
        transformed_ticks0 = [token0_pos["bottomTick"], token0_pos["topTick"]]
    elif token1_pos:
        transformed_ticks1 = [token1_pos["bottomTick"], token1_pos["topTick"]]
    else:
        raise ValueError("Neither token0Pos nor token1Pos is available")


    # Plot token0Pos if available
    if transformed_ticks0:
        pos = token0_pos
        ax.plot([transformed_ticks0[0], transformed_ticks0[1]],
                [pos['liquidity'], pos['liquidity']],
                color='blue', linewidth=3, label="token0Pos")

        ax.fill_between(np.array([float(transformed_ticks0[0]), float(transformed_ticks0[1])], dtype=np.float64),
                np.array([float(pos['liquidity']), float(pos['liquidity'])], dtype=np.float64),
                color='blue', alpha=0.1, label="_nolegend_")

        # Add text inside the filled area
        ax.text((transformed_ticks0[0] + transformed_ticks0[1]) / 2,
                pos['liquidity'] / 2,
                f"""
                T0: {pos['amount0'] / 10**decimals0:.4f}
                T1: {pos['amount1'] / 10**decimals1:.4f}
                """,
                color='blue', fontsize=12, ha='center', va='bottom', fontweight='bold')

        tick_values.update([transformed_ticks0[0], transformed_ticks0[1]])
        tick_positions.update([pos['bottomTick'], pos['topTick']])

    # Plot token1Pos if available
    if transformed_ticks1:
        pos = token1_pos
        ax.plot([transformed_ticks1[0], transformed_ticks1[1]],
                [pos['liquidity'], pos['liquidity']],
                color='orange', linewidth=3, label="token1Pos")

        ax.fill_between(np.array([float(transformed_ticks1[0]), float(transformed_ticks1[1])], dtype=np.float64),
                np.array([float(pos['liquidity']), float(pos['liquidity'])], dtype=np.float64),
                color='orange', alpha=0.1, label="_nolegend_")

        ax.text((transformed_ticks1[0] + transformed_ticks1[1]) / 2,
                pos['liquidity'] / 2,
                f"""
                T0: {pos['amount0'] / 10**decimals0:.4f}
                T1: {pos['amount1'] / 10**decimals1:.4f}
                """,
                color='orange', fontsize=12, ha='center', va='bottom', fontweight='bold')

        tick_values.update([transformed_ticks1[0], transformed_ticks1[1]])
        tick_positions.update([pos['bottomTick'], pos['topTick']])

    # Sort tick positions
    tick_positions = sorted(tick_positions)
    tick_values = sorted(tick_values)

    ax.set_xticks(list(tick_values))
    ax.set_xticklabels(tick_positions)  # Keep original labels

    # Labels and title
    ax.set_xlabel("Tick (transformed scale)")
    ax.set_ylabel("Liquidity")
    ax.set_title(f"Liquidity Distribution")
    ax.legend(["USDC", "ETH"])

    return fig

def main():
    # Извлекаем параметр 'vault' (если он есть)
    vault_id = st.query_params.get('vault', '')
    
    vault, pool = get_contracts(vault_id)
    
    # vault_id = "0x1487d907247e6e1bcfb6c73b193c74a16266368c"  # ID волта
    # pool_id = "0xabff72aee1ba72fc459acd5222dd84a3182411bb"
    
	    # Фильтрация по последнему месяцу
    one_month_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)

    all_events, rebalances_only = get_all_vault_events(client, vault_id)
    filtered_events = [e for e in all_events if int(
        e['createdAtTimestamp']) >= one_month_ago.timestamp()]

    price_changes = process_swaps(sorted(get_pool_swaps(client_analytics, pool.address.lower()), key=lambda swap: swap['timestamp']))
    
    st.title("ALM Vault Dashboard: WETH/USDC")

    vault_data = get_vault_data(vault_id)
    if not vault_data:
        st.error("Vault data not found.")
        return

    if not filtered_events:
        st.error("No recent updates found.")
        return

    decimals0 = int(vault_data['decimals0'])
    decimals1 = int(vault_data['decimals1'])
    df = process_update_data(filtered_events, decimals0, decimals1)

    # eth       usdc
    base_amount0, limit_amount0, base_amount1, limit_amount1, base_liquidity, limit_liquidity = get_positions_amounts(vault)
    reserve0 = base_amount0 + limit_amount0
    reserve1 = base_amount1 + limit_amount1
    price, tick = get_pool_price(pool)
    
    base_lower, base_upper, limit_lower, limit_upper = get_pos_ticks(vault)

    reserve0_in_token1 = reserve0 * price

    reserve0_usd = reserve0_in_token1 / (10 ** 6)
    reserve1_usd = reserve1 / (10 ** 6)

    tvl = reserve0_usd + reserve1_usd

    st.subheader("Current Vault Metrics")
    current_tvl = tvl
    current_ratio = reserve1_usd / tvl * 100
    deposit_token_inventory = reserve1_usd
    paired_token_inventory = reserve0_usd
    holders = vault_data['holdersCount']

    row1 = st.columns(3)
    row2 = st.columns(2)

    row1[0].metric("TVL (USD)", f"${current_tvl:,.2f}")
    row1[1].metric("Deposit Token Ratio", f"{current_ratio:.2f}%")
    row1[2].metric("Deposit Token Inventory",
                   f"${deposit_token_inventory:,.2f}")
    row2[0].metric("Paired Token Inventory", f"${paired_token_inventory:,.2f}")
    row2[1].metric("Total Holders", holders)

    if rebalances_only:
        last_rebalances = sorted(rebalances_only, key=lambda r: int(r['createdAtTimestamp']), reverse=True)[:5]
        df_rebalances = process_rebalance_data(last_rebalances, decimals0, decimals1)

        st.subheader("Last 5 Rebalances")
        st.dataframe(df_rebalances, use_container_width=True)
    else:
        st.warning("No rebalances found.")
    
    st.subheader("Current ALM Position Ranges")
    fig = plot_positions(
        token0_pos={
            'bottomTick': base_lower,
            'topTick': base_upper,
            'liquidity': base_liquidity,
            'amount0': base_amount0,
            'amount1': base_amount1
        },
        token1_pos={
            'bottomTick': limit_lower,
            'topTick': limit_upper,
            'liquidity': limit_liquidity,
            'amount0': limit_amount0,
            'amount1': limit_amount1
        },
        decimals0=decimals0,
        decimals1=decimals1
    )
    st.pyplot(fig)

    st.subheader("TVL Over Time")
    fig_tvl = px.line(df, x='date', y='TVL', title='TVL Over Time (USD)')
    st.plotly_chart(fig_tvl, use_container_width=True)

    st.subheader("Total Fees Earned Per Day")
    fig_fees = px.bar(df, x='date', y='Fees',
                    title='Total Fees Earned Per Day (USD)')
    st.plotly_chart(fig_fees, use_container_width=True)

    st.subheader("Historical Deposit Token Ratio")
    fig_ratio = px.line(df, x='date', y='Deposit Token Ratio (%)',
                        title='Deposit Token Ratio Over Time (%)')
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.subheader("Price change over time")
    fig_tvl = px.line(price_changes, x='date', y='Price change (%)', title='Price change (%)')
    st.plotly_chart(fig_tvl, use_container_width=True)

    # plot_tvl(filtered_events)


if __name__ == "__main__":
    main()
