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

    # Order configuration
    order_refresh_time = 5  # Update every 5 seconds
    order_levels = 5  # 5 levels of orders on each side
    base_order_amount = 5  # Base HEI amount per level
    order_amount_scaling = 1.5  # Each level increases by 50%

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

    # Price source
    price_source = PriceType.MidPrice

    # Candles configuration
    candles_interval = "1s"  # 1 second candles for best bid/ask tracking
    eth_candles_interval = "1m"  # 1 minute for ETH RSI
    max_records = 1000

    # Internal state
    create_timestamp = 0
    last_arb_check = 0
    arb_check_interval = 10  # Check arbitrage every 2 seconds

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

        # Track filled orders for arbitrage
        self.pending_hedges = []

    markets = {exchange: {maker_pair, taker_pair, eth_pair}}

    def on_stop(self):
        """Stop candles when strategy stops"""
        self.hei_candles.stop()
        self.eth_candles.stop()

    def on_tick(self):
        """Main strategy logic executed on each tick"""
        # Update market making orders
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

        # Check for arbitrage opportunities
        if self.last_arb_check + self.arb_check_interval <= self.current_timestamp:
            self.check_arbitrage_opportunity()
            self.last_arb_check = self.current_timestamp

        # Process pending hedges
        self.process_pending_hedges()

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
                # First level matches best bid/ask
                buy_price = best_bid
                sell_price = best_ask
            else:
                # Other levels use spreads from reference
                ref_price = self.connectors[self.exchange].get_price_by_type(self.maker_pair, self.price_source)
                buy_price = ref_price * (Decimal("1") - level_bid_spread)
                sell_price = ref_price * (Decimal("1") + level_ask_spread)

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

    def check_arbitrage_opportunity(self):
        """Check for arbitrage opportunities between HEI/BTC and HEI/USDT"""
        try:
            # Get HEI/BTC prices
            hei_btc_bid = self.connectors[self.exchange].get_price(self.maker_pair, False)
            hei_btc_ask = self.connectors[self.exchange].get_price(self.maker_pair, True)

            # Get HEI/USDT prices
            hei_usdt_bid = self.connectors[self.exchange].get_price(self.taker_pair, False)
            hei_usdt_ask = self.connectors[self.exchange].get_price(self.taker_pair, True)

            # Get BTC/USDT price for conversion
            btc_usdt = self.connectors[self.exchange].get_price_by_type("BTC-USDT", PriceType.MidPrice)

            # Convert HEI/BTC prices to USDT equivalent
            hei_btc_bid_usdt = hei_btc_bid * btc_usdt
            hei_btc_ask_usdt = hei_btc_ask * btc_usdt

            # Check arbitrage: Buy HEI/USDT, Sell HEI/BTC
            arb_spread_1 = (hei_btc_bid_usdt - hei_usdt_ask) / hei_usdt_ask * 10000  # in bps

            # Check arbitrage: Buy HEI/BTC, Sell HEI/USDT
            arb_spread_2 = (hei_usdt_bid - hei_btc_ask_usdt) / hei_btc_ask_usdt * 10000  # in bps

            if arb_spread_1 > self.min_arb_spread:
                self.logger().info(f"Arbitrage opportunity: Buy HEI/USDT at {hei_usdt_ask}, Sell HEI/BTC at {hei_btc_bid} (equiv {hei_btc_bid_usdt})")
                self.execute_arbitrage(TradeType.BUY, self.taker_pair, TradeType.SELL, self.maker_pair)

            elif arb_spread_2 > self.min_arb_spread:
                self.logger().info(f"Arbitrage opportunity: Buy HEI/BTC at {hei_btc_ask} (equiv {hei_btc_ask_usdt}), Sell HEI/USDT at {hei_usdt_bid}")
                self.execute_arbitrage(TradeType.BUY, self.maker_pair, TradeType.SELL, self.taker_pair)

        except Exception as e:
            self.logger().error(f"Error checking arbitrage: {e}")

    def execute_arbitrage(self, side1: TradeType, pair1: str, side2: TradeType, pair2: str):
        """Execute arbitrage trade"""
        amount = Decimal(str(self.arb_order_amount))

        # Place first order (taker)
        if side1 == TradeType.BUY:
            self.buy(self.exchange, pair1, amount, OrderType.MARKET)
        else:
            self.sell(self.exchange, pair1, amount, OrderType.MARKET)

        # Queue the hedge order
        self.pending_hedges.append({
            "side": side2,
            "pair": pair2,
            "amount": amount,
            "timestamp": self.current_timestamp
        })

    def process_pending_hedges(self):
        """Process pending hedge orders from arbitrage"""
        completed_hedges = []

        for hedge in self.pending_hedges:
            # Wait a bit before placing hedge to ensure first order is filled
            if self.current_timestamp - hedge["timestamp"] > 0.5:
                if hedge["side"] == TradeType.BUY:
                    self.buy(self.exchange, hedge["pair"], hedge["amount"], OrderType.MARKET)
                else:
                    self.sell(self.exchange, hedge["pair"], hedge["amount"], OrderType.MARKET)
                completed_hedges.append(hedge)

        # Remove completed hedges
        for hedge in completed_hedges:
            self.pending_hedges.remove(hedge)

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

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle filled orders"""
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} at {round(event.price, 8)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

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

            # Arbitrage Opportunities
            btc_usdt = self.connectors[self.exchange].get_price("BTC-USDT", PriceType.MidPrice)
            hei_btc_bid_usdt = hei_btc_bid * btc_usdt
            hei_btc_ask_usdt = hei_btc_ask * btc_usdt

            arb_spread_1 = (hei_btc_bid_usdt - hei_usdt_ask) / hei_usdt_ask * 10000
            arb_spread_2 = (hei_usdt_bid - hei_btc_ask_usdt) / hei_btc_ask_usdt * 10000

            lines.extend([
                f"\n🔄 ARBITRAGE OPPORTUNITIES:",
                f"  Buy USDT, Sell BTC: {arb_spread_1:.1f} bps {'✅' if arb_spread_1 > self.min_arb_spread else '❌'}",
                f"  Buy BTC, Sell USDT: {arb_spread_2:.1f} bps {'✅' if arb_spread_2 > self.min_arb_spread else '❌'}"
            ])
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
            f"  Refresh Time: {self.order_refresh_time}s",
            f"  Arbitrage Amount: {self.arb_order_amount} HEI",
            f"  Pending Hedges: {len(self.pending_hedges)}"
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
