# Testnet E2E テスト最終確認手順

## 前提条件

`.env` に以下が設定済みであること:

```
BYBIT_TESTNET_API_KEY=your_testnet_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_secret_here
```

キーの取得先: https://testnet.bybit.com/app/user/api-management

## 実行手順

### Step 1: Testnet E2E テストの完全実行

```bash
# Skipされていたテストを含めて全実行
pytest tests/test_live_execution.py -v
```

期待結果: 8 passed, 0 skipped

### Step 2: エラー発生時の対処

| エラー | 対処 |
|--------|------|
| 署名エラー (10004) | `.env` のキーをダブルクォートなしで再設定 |
| 残高不足 (130021) | Testnet faucet で USDT を取得 |
| シンボル不一致 | Testnet で BTCUSDT が利用可能か確認 |
| PostOnly rejected (140025) | 指値を best bid/ask から十分離す |

### Step 3: テスト後のクリーンアップ確認

```bash
# テスト後に残存注文・ポジションがないことを確認
python3 -c "
from src.exchange_api import BybitClient
client = BybitClient(testnet=True)

# 安全確認: testnet であること
assert client.testnet, 'MUST be testnet'
assert 'testnet' in client.base_url

# オープン注文確認
orders = client.get_open_orders('BTCUSDT')
open_list = orders.get('result', {}).get('list', [])
print(f'Open orders: {len(open_list)}')
for o in open_list:
    print(f'  {o[\"side\"]} {o[\"qty\"]} @ {o[\"price\"]} ({o[\"orderStatus\"]})')

# ポジション確認
positions = client.get_positions('BTCUSDT')
pos_list = positions.get('result', {}).get('list', [])
for p in pos_list:
    size = float(p.get('size', 0))
    if size > 0:
        print(f'  Position: {p[\"side\"]} {size} @ {p[\"avgPrice\"]}')
    else:
        print(f'  No open position')

# 残存があれば全キャンセル
if open_list:
    print('Cancelling all...')
    result = client.cancel_all_orders('BTCUSDT')
    print(f'Cancel result: {result.success}')
"
```

### Step 4: Order Manager 統合テスト (手動)

```bash
# Order Manager の Taker Fallback を手動で確認
python3 -c "
import asyncio
from src.exchange_api import BybitClient
from src.order_manager import OrderManager

async def test_fallback():
    client = BybitClient(testnet=True)
    assert client.testnet

    om = OrderManager(client)
    await om.start()

    # 1. 約定しない価格で Maker 発注
    import requests
    price_resp = requests.get(
        f'{client.base_url}/v5/market/tickers',
        params={'category': 'linear', 'symbol': 'BTCUSDT'},
        timeout=5,
    ).json()
    last = float(price_resp['result']['list'][0]['lastPrice'])
    far_price = round(last * 0.85, 2)  # 15% below

    order = await om.place_maker_order('BTCUSDT', 'Buy', 0.001, far_price)
    print(f'Placed: {order.order_link_id} state={order.state}')

    # 2. Taker fallback (5秒 timeout → Cancel → IOC)
    result = await om.wait_for_fill(order.order_link_id, timeout_s=5.0, taker_fallback=True)
    print(f'Final state: {result.state}')
    print(f'Fill price: {result.filled_price}, Maker: {result.is_maker_fill}')

    await om.stop()

asyncio.run(test_fallback())
"
```

## 安全確認チェックリスト

- [ ] `.env` に BYBIT_TESTNET_API_KEY が設定されている
- [ ] テストコード内で `assert client.testnet is True` が通る
- [ ] テストコード内で `assert "testnet" in client.base_url` が通る
- [ ] 全8テストが PASS
- [ ] テスト後にオープン注文が 0
- [ ] テスト後にオープンポジションが 0
- [ ] Paper Trader (PID確認) が影響を受けていない

## 制約事項

- **Mainnet APIキーは絶対に使用しない**
- 稼働中の Paper Trader には干渉しない
- テスト完了後は必ずクリーンアップ確認を実行する
