"""
Trade Logger - Per-transactie logging voor RL trading agents.

Logt elke individuele buy/sell transactie naar een CSV bestand met alle
relevante financiële details: ingezet bedrag, hoeveelheid BTC, prijs,
fees, winst/verlies, portfolio waarde.

Gebruik:
    trade_logger = TradeLogger(log_dir)
    trade_logger.log_trade(episode, step, trade_info, balance, btc_held, portfolio_value)
    trades_df = trade_logger.get_trades()  # pandas DataFrame of list of dicts
"""

import os
import csv
from typing import Dict, Any, List, Optional


TRADE_CSV_COLUMNS = [
    'episode',
    'step',
    'action',
    'btc_amount',
    'btc_price',
    'usd_value',
    'fee',
    'profit_loss',
    'balance_before',
    'balance_after',
    'btc_held_before',
    'btc_held_after',
    'portfolio_value',
    'cumulative_pnl',
]


class TradeLogger:
    """
    Per-transactie logger die elke buy/sell bijhoudt in een CSV.
    
    Attributes:
        log_path: Pad naar het trade CSV bestand
        trades: Lijst van alle gelogde trades (in-memory)
        cumulative_pnl: Cumulatieve winst/verlies over alle trades
    """
    
    def __init__(self, log_dir: str, filename: str = 'trade_log.csv'):
        """
        Args:
            log_dir: Directory waar het CSV bestand wordt opgeslagen
            filename: Naam van het CSV bestand
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.trades: List[Dict[str, Any]] = []
        self.cumulative_pnl = 0.0
        self._pending: List[Dict[str, Any]] = []
        self._file_exists = os.path.exists(self.log_path)

        # Maak CSV header als bestand nog niet bestaat
        if not self._file_exists:
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(TRADE_CSV_COLUMNS)
            self._file_exists = True
    
    def log_trade(
        self,
        episode: int,
        step: int,
        trade_info: Dict[str, Any],
        balance_before: float,
        balance_after: float,
        btc_held_before: float,
        btc_held_after: float,
        portfolio_value: float
    ):
        """
        Log een enkele trade (buy of sell) naar CSV en memory.
        
        Args:
            episode: Huidig episode nummer
            step: Huidige training step
            trade_info: Dict van trading env met keys:
                        action, executed, type, amount, price, fee, profit (sells)
            balance_before: USD balans voor de trade
            balance_after: USD balans na de trade
            btc_held_before: BTC positie voor de trade
            btc_held_after: BTC positie na de trade
            portfolio_value: Totale portfolio waarde na de trade
        """
        if not trade_info.get('executed', False):
            return
        
        action_type = trade_info.get('type', 'unknown')
        btc_amount = trade_info.get('amount', 0.0)
        btc_price = trade_info.get('price', 0.0)
        fee = trade_info.get('fee', 0.0)
        
        # Bereken USD waarde van de trade
        usd_value = btc_amount * btc_price
        
        # Profit/loss: voor sells komt dit uit trade_info, voor buys is het 0
        profit_loss = trade_info.get('profit', 0.0) if action_type == 'sell' else 0.0
        
        # Update cumulatieve PnL (alleen bij sells)
        self.cumulative_pnl += profit_loss
        
        trade_record = {
            'episode': episode,
            'step': step,
            'action': action_type,
            'btc_amount': btc_amount,
            'btc_price': btc_price,
            'usd_value': usd_value,
            'fee': fee,
            'profit_loss': profit_loss,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'btc_held_before': btc_held_before,
            'btc_held_after': btc_held_after,
            'portfolio_value': portfolio_value,
            'cumulative_pnl': self.cumulative_pnl,
        }
        
        self.trades.append(trade_record)
        self._pending.append(trade_record)

    def flush_to_csv(self):
        """Schrijf gebufferde trades naar CSV en leeg de buffer."""
        if not self._pending:
            return
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for trade_record in self._pending:
                writer.writerow([
                    trade_record['episode'],
                    trade_record['step'],
                    trade_record['action'],
                    f"{trade_record['btc_amount']:.8f}",
                    f"{trade_record['btc_price']:.2f}",
                    f"{trade_record['usd_value']:.2f}",
                    f"{trade_record['fee']:.4f}",
                    f"{trade_record['profit_loss']:.4f}",
                    f"{trade_record['balance_before']:.2f}",
                    f"{trade_record['balance_after']:.2f}",
                    f"{trade_record['btc_held_before']:.8f}",
                    f"{trade_record['btc_held_after']:.8f}",
                    f"{trade_record['portfolio_value']:.2f}",
                    f"{trade_record['cumulative_pnl']:.4f}",
                ])
        self._pending.clear()
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Return alle trades als lijst van dicts."""
        return self.trades
    
    def get_sell_profits(self) -> List[float]:
        """Return lijst van profit/loss voor elke sell trade."""
        return [t['profit_loss'] for t in self.trades if t['action'] == 'sell']
    
    def get_trade_sizes(self) -> List[float]:
        """Return lijst van USD bedragen per trade (buys en sells)."""
        return [t['usd_value'] for t in self.trades]
    
    def get_buy_sizes(self) -> List[float]:
        """Return lijst van ingezette USD bedragen per buy."""
        return [t['usd_value'] for t in self.trades if t['action'] == 'buy']
    
    def get_sell_sizes(self) -> List[float]:
        """Return lijst van verkochte USD bedragen per sell."""
        return [t['usd_value'] for t in self.trades if t['action'] == 'sell']
    
    def get_cumulative_pnl_series(self) -> List[float]:
        """Return cumulatieve PnL na elke sell trade."""
        result = []
        cum = 0.0
        for t in self.trades:
            if t['action'] == 'sell':
                cum += t['profit_loss']
                result.append(cum)
        return result
    
    def get_portfolio_values_at_trades(self) -> List[float]:
        """Return portfolio waarde op elk trade moment."""
        return [t['portfolio_value'] for t in self.trades]
    
    def get_summary(self) -> Dict[str, Any]:
        """Geeft een samenvatting van alle trades."""
        sells = [t for t in self.trades if t['action'] == 'sell']
        buys = [t for t in self.trades if t['action'] == 'buy']
        
        winning_sells = [t for t in sells if t['profit_loss'] > 0]
        losing_sells = [t for t in sells if t['profit_loss'] <= 0]
        
        total_profit = sum(t['profit_loss'] for t in winning_sells)
        total_loss = sum(abs(t['profit_loss']) for t in losing_sells)
        total_fees = sum(t['fee'] for t in self.trades)
        
        avg_buy_size = sum(t['usd_value'] for t in buys) / max(len(buys), 1)
        avg_sell_size = sum(t['usd_value'] for t in sells) / max(len(sells), 1)
        
        return {
            'total_trades': len(self.trades),
            'total_buys': len(buys),
            'total_sells': len(sells),
            'winning_sells': len(winning_sells),
            'losing_sells': len(losing_sells),
            'win_rate': len(winning_sells) / max(len(sells), 1),
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': total_profit - total_loss,
            'total_fees': total_fees,
            'avg_buy_size_usd': avg_buy_size,
            'avg_sell_size_usd': avg_sell_size,
            'avg_profit_per_win': total_profit / max(len(winning_sells), 1),
            'avg_loss_per_loss': total_loss / max(len(losing_sells), 1),
            'cumulative_pnl': self.cumulative_pnl,
        }
    
    def restore_from_list(self, trades: List[Dict[str, Any]]):
        """Herstel trade state vanuit een checkpoint."""
        self.trades = trades
        self.cumulative_pnl = trades[-1]['cumulative_pnl'] if trades else 0.0
    
    def print_summary(self):
        """Print een mooie samenvatting naar stdout."""
        s = self.get_summary()
        print(f"\n{'='*60}")
        print(f"  TRADE SUMMARY")
        print(f"{'='*60}")
        print(f"  Total trades:      {s['total_trades']:,}")
        print(f"  Buys:              {s['total_buys']:,}")
        print(f"  Sells:             {s['total_sells']:,}")
        print(f"  Winning sells:     {s['winning_sells']:,} ({s['win_rate']*100:.1f}%)")
        print(f"  Losing sells:      {s['losing_sells']:,}")
        print(f"  ---")
        print(f"  Total profit:      ${s['total_profit']:,.2f}")
        print(f"  Total loss:        ${s['total_loss']:,.2f}")
        print(f"  Net PnL:           ${s['net_pnl']:+,.2f}")
        print(f"  Total fees:        ${s['total_fees']:,.2f}")
        print(f"  ---")
        print(f"  Avg buy size:      ${s['avg_buy_size_usd']:,.2f}")
        print(f"  Avg sell size:     ${s['avg_sell_size_usd']:,.2f}")
        print(f"  Avg profit/win:    ${s['avg_profit_per_win']:,.2f}")
        print(f"  Avg loss/loss:     ${s['avg_loss_per_loss']:,.2f}")
        print(f"{'='*60}\n")
