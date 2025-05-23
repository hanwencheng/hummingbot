you are a crypto trading expert and a professional software developer.

I holds lots of HEI, USDT, and BTC. Now I want to:
1. boost the trading volume of HEI/BTC pair in Binance
2. find arbitrage opportunities between HEI/USDT and HEI/USDC and HEI/BTC
3. I want to accumulate the HEI token without big price impact.

Please read through the scripts in scripts/heima folder, and think a strategy that:
1. Use real binance exchange connector
2. according to simple_pmm.py, Place 5 level positions in both buy and sell for market making in HEI/BTC, update the position every 5 seconds.
3. Use the ETH/USDT's RSI as trend indicator to adjust the order, if bullish, we should increase HEI buyer spread, but leave the sell order unchanged
4. refer to the candle data usage in pmm-volatility-spread.py, use the 1s candle stick to make sure my order has the same price with best bid order and best ask order. 
5. refer to the cross-exchange arbitrage usage in simple_xemm.py, create the arbitrage code to use HEI/BTC as the maker market, and use the HEI/USDT as the taker market.
6. override the format_status function to better illustrate all the information, with candle data information displayed, but prevent to save the candle data information in the disk
