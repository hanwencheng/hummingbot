import logging
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig


class HEITradingStrategy(ScriptStrategyBase):
    """
    Advanced HEI Trading Strategy combining:
    1. 5-level market making on HEI/BTC
    2. RSI-based trend adjustments using ETH/USDT
    3. Cross-exchange arbitrage between HEI/BTC and HEI/USDT
    4. Volume boosting with minimal price impact
    """

    # Configuration
    exchange = "binance"  # Real Binance exchange
    maker_pair = "HEI-BTC"
    taker_pair = "HEI-USDT"
    eth_pair = "ETH-USDT"
    btc_pair = "BTC-USDT"

    # Order configuration
    order_refresh_time = 10  # Update every 5 seconds
    order_levels = 5  # 5 levels of orders on each side
    base_order_amount = 50  # Base HEI amount per level
    order_amount_scaling = 1.2  # Each level increases by 50%

    # Spread configuration (in basis points)
    base_bid_spread = 20  # 0.1%
    base_ask_spread = 20  # 0.1%
    spread_increment = 20  # 0.05% increment per level

    # RSI configuration
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    rsi_adjustment_factor = 2.0  # Multiply bid spread by this when bullish

    # Arbitrage configuration
    min_arb_spread = 20  # Minimum 0.2% spread for arbitrage
    arb_order_amount = 100  # HEI amount for arbitrage trades
    arbitrage_threshold = 40

    # Price source
    price_source = PriceType.LastTrade

    # Candles configuration
    candles_interval = "1s"  # 1 second candles for best bid/ask tracking
    eth_candles_interval = "1m"  # 1 minute for ETH RSI
    max_records = 1000

    # Internal state
    create_timestamp = 0

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        # Initialize candles for HEI/BTC (1s for order book tracking)
        self.hei_candles = CandlesFactory.get_candle(
            CandlesConfig(
                connector=self.exchange,
                trading_pair=self.maker_pair,
                interval=self.candles_interval,
                max_records=self.max_records
            )
        )

        # Initialize candles for ETH/USDT (1m for RSI)
        self.eth_candles = CandlesFactory.get_candle(
            CandlesConfig(
                connector=self.exchange,
                trading_pair=self.eth_pair,
                interval=self.eth_candles_interval,
                max_records=self.max_records
            )
        )

        # Start candles
        self.hei_candles.start()
        self.eth_candles.start()

    markets = {exchange: {maker_pair, taker_pair, eth_pair, btc_pair}}

    async def on_stop(self):
        """Stop strategy and clean up"""
        # Stop candles
        self.hei_candles.stop()
        self.eth_candles.stop()

        self.logger().info("HEI Trading Strategy stopped and cleaned up")

    def on_tick(self):
        """Main strategy logic executed on each tick"""
        # Check if strategy is ready to trade
        if not self.ready_to_trade:
            return

        # Update market making orders
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def get_eth_rsi(self) -> Optional[float]:
        """Calculate RSI for ETH/USDT"""
        try:
            candles_df = self.eth_candles.candles_df
            if len(candles_df) < self.rsi_period + 1:
                return None

            # Add RSI indicator
            candles_df.ta.rsi(length=self.rsi_period, append=True)
            return candles_df[f"RSI_{self.rsi_period}"].iloc[-1]
        except Exception as e:
            self.logger().error(f"Error calculating RSI: {e}")
            return None

    def get_adjusted_spreads(self) -> tuple:
        """Get spreads adjusted for market conditions"""
        bid_spread = self.base_bid_spread / 10000  # Convert from bps
        ask_spread = self.base_ask_spread / 10000

        # Adjust based on RSI
        rsi = self.get_eth_rsi()
        if rsi is not None:
            if rsi > 50:  # Bullish market
                # Increase bid spread (buy lower) when market is bullish
                bid_spread *= self.rsi_adjustment_factor
                # Keep ask spread unchanged to accumulate HEI
            elif rsi < 50:  # Bearish market
                # More aggressive buying in bearish conditions
                bid_spread *= 0.6

        return Decimal(str(bid_spread)), Decimal(str(ask_spread))

    def get_best_prices(self) -> tuple:
        """Get best bid and ask prices from order book"""
        try:
            best_bid = self.connectors[self.exchange].get_price(self.maker_pair, False)
            best_ask = self.connectors[self.exchange].get_price(self.maker_pair, True)
            return best_bid, best_ask
        except Exception as e:
            self.logger().error(f"Error getting best prices: {e}")
            # Fallback to mid price
            mid_price = self.connectors[self.exchange].get_mid_price(self.maker_pair)
            return mid_price * Decimal("0.999"), mid_price * Decimal("1.001")

    def create_proposal(self) -> List[OrderCandidate]:
        """Create order proposal with 5 levels on both sides"""
        # Return empty list if not ready to trade
        if not self.ready_to_trade:
            return []

        best_bid, best_ask = self.get_best_prices()
        base_bid_spread, base_ask_spread = self.get_adjusted_spreads()

        buy_orders = []
        sell_orders = []

        for level in range(self.order_levels):
            # Calculate spread for this level
            level_bid_spread = base_bid_spread + Decimal(str(level * self.spread_increment / 10000))
            level_ask_spread = base_ask_spread + Decimal(str(level * self.spread_increment / 10000))

            # Calculate order amount for this level
            order_amount = Decimal(str(self.base_order_amount * (self.order_amount_scaling ** level)))

            # Calculate prices - match best bid/ask for first level
            if level == 0:
                ref_price = self.connectors[self.exchange].get_price_by_type(self.maker_pair, self.price_source)
                target_buy_price = ref_price * (Decimal("1") - level_bid_spread)
                target_sell_price = ref_price * (Decimal("1") + level_ask_spread)

                # Log the best price and target price
                self.logger().info(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
                self.logger().info(f"Target Buy Price: {target_buy_price}, Target Sell Price: {target_sell_price}")

                # First level matches best bid/ask
                buy_price = min(best_bid, target_buy_price)
                sell_price = max(best_ask, target_sell_price)
            else:
                # Other levels use spreads from the last level plus spread
                buy_price = buy_orders[-1].price * (Decimal("1") - level_bid_spread)
                sell_price = sell_orders[-1].price * (Decimal("1") + level_ask_spread)

            # Create order candidates
            buy_order = OrderCandidate(
                trading_pair=self.maker_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=order_amount,
                price=buy_price
            )

            sell_order = OrderCandidate(
                trading_pair=self.maker_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=order_amount,
                price=sell_price
            )

            buy_orders.append(buy_order)
            sell_orders.append(sell_order)

        return buy_orders + sell_orders

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust proposal to available budget"""
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place the adjusted orders"""
        for order in proposal:
            if order.amount > 0:
                self.place_order(connector_name=self.exchange, order=order)



    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place an order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def is_active_maker_order(self, event: OrderFilledEvent):
        """
        Helper function that checks if order is an active order on the maker exchange
        """
        return event.trading_pair == self.maker_pair

    def get_sell_price_in_usdt(self, event: OrderFilledEvent):
        """Calculate the USD value of an order filled event."""
        btc_usdt_price = self.connectors[self.exchange].get_price(self.btc_pair, PriceType.MidPrice)
        return event.price * btc_usdt_price

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle filled orders and execute arbitrage"""
        if event.trade_type == TradeType.BUY and self.is_active_maker_order(event):
            self.logger().info(f"Filled maker buy order at price {event.price:.6f} for amount {event.amount:.2f}")
            self.place_sell_order(self.config.taker_pair, event.amount)
        else:
            if event.trade_type == TradeType.SELL and self.is_active_maker_order(event):
                self.logger().info(f"Filled maker sell order at price {event.price:.6f} for amount {event.amount:.2f}")
                self.place_buy_order(self.config.taker_pair, event.amount)

    def place_buy_taker_order(self, event: OrderFilledEvent, amount: Decimal, order_type: OrderType = OrderType.LIMIT):
        buy_price = self.connectors[self.exchange].get_price_for_volume(self.taker_pair, True, amount)
        is_profitable = (self.get_sell_price_in_usdt(event) - buy_price) / buy_price > self.arbitrage_threshold/10000
        if is_profitable:
            buy_order = OrderCandidate(self.taker_pair, is_maker=False, order_type=order_type, order_side=TradeType.BUY, amount=amount, price=buy_price)
            buy_order_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidate(buy_order, all_or_none=False)
            self.buy(self.exchange, self.taker_pair, buy_order_adjusted.amount, buy_order_adjusted.order_type, buy_order_adjusted.price)

    def place_sell_taker_order(self, event: OrderFilledEvent, amount: Decimal, order_type: OrderType = OrderType.LIMIT):
        sell_price = self.connectors[self.exchange].get_price_for_volume(self.taker_pair, False, amount)
        is_profitable = (sell_price - self.get_sell_price_in_usdt(event)) / sell_price > self.arbitrage_threshold/10000
        if is_profitable:
            sell_order = OrderCandidate(self.taker_pair, is_maker=False, order_type=order_type, order_side=TradeType.SELL, amount=amount, price=sell_price)
            sell_order_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidate(sell_order, all_or_none=False)
            self.sell(self.exchange, self.taker_pair, sell_order_adjusted.amount, sell_order_adjusted.order_type, sell_order_adjusted.price)

    def format_status(self) -> str:
        """Format strategy status with comprehensive information"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        # Header
        lines.extend([
            "\n════════════════════════════════════════════════════════════════",
            "                    HEI TRADING STRATEGY STATUS                   ",
            "════════════════════════════════════════════════════════════════\n"
        ])

        # Balances
        balance_df = self.get_balance_df()
        lines.extend(["📊 BALANCES:"] + ["  " + line for line in balance_df.to_string(index=False).split("\n")])

        # Market Conditions
        lines.extend(["\n📈 MARKET CONDITIONS:"])

        # HEI Selling Price
        hei_selling_price = self.get_hei_selling_price(amount=Decimal("1"))
        if hei_selling_price is not None:
            lines.append(f"  HEI Selling Price for 1 HEI: {hei_selling_price:.6f} USDT")
        else:
            lines.append("  HEI Selling Price: Unable to fetch price")

        # RSI Status
        rsi = self.get_eth_rsi()
        if rsi is not None:
            rsi_status = "🔴 Overbought" if rsi > self.rsi_overbought else "🟢 Oversold" if rsi < self.rsi_oversold else "🟡 Neutral"
            lines.append(f"  ETH/USDT RSI: {rsi:.2f} {rsi_status}")

        # Spread Adjustments
        base_bid_spread, base_ask_spread = self.get_adjusted_spreads()
        lines.extend([
            f"  Adjusted Bid Spread: {base_bid_spread * 10000:.1f} bps",
            f"  Adjusted Ask Spread: {base_ask_spread * 10000:.1f} bps"
        ])

        # Price Information
        lines.extend(["\n💹 PRICE INFORMATION:"])
        try:
            # HEI/BTC
            hei_btc_mid = self.connectors[self.exchange].get_mid_price(self.maker_pair)
            hei_btc_bid, hei_btc_ask = self.get_best_prices()
            lines.extend([
                f"  HEI/BTC:",
                f"    Mid: {hei_btc_mid:.8f} | Bid: {hei_btc_bid:.8f} | Ask: {hei_btc_ask:.8f}",
                f"    Spread: {(hei_btc_ask - hei_btc_bid) / hei_btc_mid * 10000:.1f} bps"
            ])

            # HEI/USDT
            hei_usdt_mid = self.connectors[self.exchange].get_mid_price(self.taker_pair)
            hei_usdt_bid = self.connectors[self.exchange].get_price(self.taker_pair, False)
            hei_usdt_ask = self.connectors[self.exchange].get_price(self.taker_pair, True)
            lines.extend([
                f"  HEI/USDT:",
                f"    Mid: {hei_usdt_mid:.4f} | Bid: {hei_usdt_bid:.4f} | Ask: {hei_usdt_ask:.4f}",
                f"    Spread: {(hei_usdt_ask - hei_usdt_bid) / hei_usdt_mid * 10000:.1f} bps"
            ])

            # Calculate potential arbitrage spreads
            try:
                # Get HEI/BTC prices
                hei_btc_bid = self.connectors[self.exchange].get_price(self.maker_pair, False)
                hei_btc_ask = self.connectors[self.exchange].get_price(self.maker_pair, True)

                # Get HEI/USDT prices
                hei_usdt_bid = self.connectors[self.exchange].get_price(self.taker_pair, False)
                hei_usdt_ask = self.connectors[self.exchange].get_price(self.taker_pair, True)

                # Arbitrage Opportunities
                btc_usdt = self.connectors[self.exchange].get_price("BTC-USDT", PriceType.MidPrice)
                hei_btc_bid_usdt = hei_btc_bid * btc_usdt
                hei_btc_ask_usdt = hei_btc_ask * btc_usdt

                # Calculate spreads
                # If we buy HEI/BTC and sell HEI/USDT (when our HEI/BTC buy order fills)
                buy_btc_sell_usdt_spread = (hei_usdt_bid - hei_btc_ask_usdt) / hei_btc_ask_usdt * 10000

                # If we sell HEI/BTC and buy HEI/USDT (when our HEI/BTC sell order fills)
                sell_btc_buy_usdt_spread = (hei_btc_bid_usdt - hei_usdt_ask) / hei_usdt_ask * 10000

                lines.extend([
                    f"\n🔄 ARBITRAGE POTENTIAL (on maker fill):",
                    f"  If BTC buy fills → Sell USDT: {buy_btc_sell_usdt_spread:.1f} bps {'✅' if buy_btc_sell_usdt_spread > 0 else '❌'}",
                    f"  If BTC sell fills → Buy USDT: {sell_btc_buy_usdt_spread:.1f} bps {'✅' if sell_btc_buy_usdt_spread > 0 else '❌'}"
                ])
            except Exception as e:
                lines.append(f"  Error calculating arbitrage: {e}")
        except Exception as e:
            lines.append(f"  Error getting prices: {e}")

        # Active Orders
        lines.extend(["\n📋 ACTIVE ORDERS:"])
        try:
            active_orders = self.get_active_orders(self.exchange)
            if active_orders:
                buy_orders = [o for o in active_orders if o.is_buy]
                sell_orders = [o for o in active_orders if not o.is_buy]

                lines.extend([
                    f"  Buy Orders: {len(buy_orders)} levels",
                    f"  Sell Orders: {len(sell_orders)} levels"
                ])

                # Order details
                if buy_orders:
                    lines.append("  Buy Levels:")
                    for i, order in enumerate(sorted(buy_orders, key=lambda x: x.price, reverse=True)[:5]):
                        lines.append(f"    L{i+1}: {order.quantity:.2f} @ {order.price:.8f}")

                if sell_orders:
                    lines.append("  Sell Levels:")
                    for i, order in enumerate(sorted(sell_orders, key=lambda x: x.price)[:5]):
                        lines.append(f"    L{i+1}: {order.quantity:.2f} @ {order.price:.8f}")
            else:
                lines.append("  No active orders")
        except Exception as e:
            lines.append(f"  Error getting orders: {e}")

        # Strategy Metrics
        lines.extend([
            f"\n⚙️  STRATEGY PARAMETERS:",
            f"  Order Levels: {self.order_levels}",
            f"  Base Amount: {self.base_order_amount} HEI",
            f"  Refresh Time: {self.order_refresh_time}s"
        ])

        # Candle Data (Recent)
        lines.extend(["\n📊 RECENT CANDLES (ETH/USDT for RSI):"])
        try:
            eth_df = self.eth_candles.candles_df.tail(5)
            if not eth_df.empty:
                eth_df['time'] = pd.to_datetime(eth_df['timestamp'], unit='ms').dt.strftime('%H:%M:%S')
                eth_df_display = eth_df[['time', 'open', 'high', 'low', 'close', 'volume']].round(2)
                for line in eth_df_display.to_string(index=False).split("\n"):
                    lines.append("  " + line)
            else:
                lines.append("  No candle data available")
        except Exception as e:
            lines.append(f"  Error displaying candles: {e}")

        lines.append("\n════════════════════════════════════════════════════════════════")

        return "\n".join(lines)
