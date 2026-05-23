"""
Crypto Trading Environment voor Reinforcement Learning
======================================================

Dit is een Gymnasium-compatibele trading environment voor BTC trading.
De agent kan kopen, verkopen of vasthouden en krijgt rewards gebaseerd
op profit/loss.

Features:
- Realistische trading simulatie met transaction costs
- Configurable position sizes en risk management
- Support voor continue en discrete action spaces
- Uitgebreide info dict voor logging

Auteur: DataDeepRL Team
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import logging

# =====================================
# LOGGING SETUP
# =====================================
# Gebruik logging voor alle debug informatie
# Level DEBUG: alle details (elke step)
# Level INFO: belangrijke events (episodes, trades)
# Level WARNING: potentiële problemen
logger = logging.getLogger(__name__)

# Performance tracking voor logging
_LOG_TRADE_DETAILS = False  # Zet op True voor verbose trade output
_LOG_STEP_INTERVAL = 1000  # Log elke N steps (voor DEBUG level)


class CryptoTradingEnv(gym.Env):
    """
    Gymnasium environment voor cryptocurrency trading.
    
    De agent observeert marktdata (OHLCV + features) en beslist om
    te kopen, verkopen of niets te doen. Het doel is om winst te maximaliseren.
    
    Observation Space:
        Box met shape (window_size, num_features) - historische features
        
    Action Space (discrete):
        0 = Hold (niets doen)
        1 = Buy (kopen)
        2 = Sell (verkopen)
        
    Action Space (continuous):
        Box(-1, 1) waar:
        -1 = Volledig verkopen
         0 = Hold
        +1 = Volledig kopen
        
    Reward:
        - Profit/loss van trades
        - Penalty voor te veel traden (transaction costs)
        - Optionele Sharpe ratio bonus
    
    Attributen:
        df: DataFrame met marktdata
        initial_balance: Startkapitaal in USDT
        transaction_fee: Fee percentage per trade (bijv. 0.001 = 0.1%)
        flat_fee: Flat fee per trade in USDT (bijv. 1.0 = $1)
        window_size: Aantal tijdstappen in observation
    """
    
    metadata = {'render_modes': ['human', 'none']}
    
    def __init__(
        self,
        df: 'pd.DataFrame' = None,
        feature_columns: List[str] = None,
        initial_balance: float = 100000.0,
        transaction_fee: float = 0.0,  # Percentage fee (bijv. 0.001 = 0.1%)
        flat_fee: float = 1.0,  # Flat fee per trade in USDT (bijv. 1.0 = $1)
        window_size: int = 100,
        max_position: float = 1.0,  # Max fractie van portfolio in BTC
        discrete_actions: bool = True,
        reward_scaling: float = 1.0,
        price_column: str = 'close',
        render_mode: str = 'none',
        # Alternative input: pre-computed sequences en prices
        sequences: np.ndarray = None,
        prices: np.ndarray = None,
        # Memory-efficient: raw features + on-the-fly windowing
        raw_features: np.ndarray = None,
        # Random start for better generalization
        random_start: bool = False,
        random_start_range: float = 0.2,  # Start within first 20% of data
        # Episode length limit (0 = unlimited, runs to end of data)
        max_episode_steps: int = 0,
    ):
        """
        Initialiseer de trading environment.
        
        Twee manieren om te initialiseren:
        1. Met DataFrame: df en feature_columns (maakt sequences intern)
        2. Met sequences: sequences en prices arrays (pre-computed)
        
        Args:
            df: DataFrame met OHLCV en feature data
            feature_columns: Lijst met kolommen om als features te gebruiken
            initial_balance: Startkapitaal in USDT
            transaction_fee: Fee percentage per trade
            window_size: Aantal historische stappen in observation
            max_position: Maximum positie grootte (0-1)
            discrete_actions: True voor discrete (buy/sell/hold), False voor continue
            reward_scaling: Schaalfactor voor rewards
            price_column: Kolom met prijzen
            render_mode: 'human' voor output, 'none' voor stil
            sequences: Pre-computed feature sequences (window_size, num_features)
            prices: Pre-computed prijzen array
            raw_features: Raw feature array (N, num_features) for memory-efficient streaming.
                          Creates sequences on-the-fly. Requires prices of same length.
            random_start: If True, start at random position for better generalization
            random_start_range: Fraction of data to sample start position from (0.0-1.0)
            max_episode_steps: Max steps per episode (0=unlimited). Creates episode boundaries.
        """
        super().__init__()
        
        # Random start parameters
        self.random_start = random_start
        self.random_start_range = random_start_range
        
        # Episode length limit
        self.max_episode_steps = max_episode_steps
        
        # Check welke input mode
        self._use_raw_features = False
        if raw_features is not None and prices is not None:
            # Mode 3: Raw features - memory efficient streaming
            # Stores only the raw (N, num_features) array.
            # Sequences are created on-the-fly via numpy slicing (zero-copy view).
            self._raw_features = raw_features
            self.num_features = raw_features.shape[1]
            self.window_size = window_size
            # Prices aligned with end of each window (same as sequences mode)
            self.prices = prices[window_size:]
            self.sequences = None
            self.df = None
            self.feature_columns = None
            self.price_column = None
            self.features = None
            self._use_sequences = False
            self._use_raw_features = True
            n_steps = len(raw_features) - window_size
            logger.info(f"Environment initialized with raw_features: {raw_features.shape} -> {n_steps:,} steps (streaming)")
        elif sequences is not None and prices is not None:
            # Mode 2: Pre-computed sequences
            self.sequences = sequences
            self._raw_features = None
            self.prices = prices
            self.num_features = sequences.shape[2] if len(sequences.shape) == 3 else sequences.shape[1]
            self.window_size = sequences.shape[1] if len(sequences.shape) == 3 else window_size
            self.df = None
            self.feature_columns = None
            self.price_column = None
            # Features zijn de sequences zelf
            self.features = sequences.reshape(len(sequences), -1)  # Flatten voor compatibility
            self._use_sequences = True
            logger.info(f"Environment initialized with sequences: {sequences.shape}")
        elif df is not None and feature_columns is not None:
            # Mode 1: DataFrame input
            self.df = df.reset_index(drop=True)
            self.feature_columns = feature_columns
            self.price_column = price_column
            self.prices = self.df[price_column].values
            self.features = self.df[feature_columns].values
            self.sequences = None
            self.num_features = len(feature_columns)
            self.window_size = window_size
            self._use_sequences = False
        else:
            raise ValueError("Provide either (df, feature_columns) or (sequences, prices)")
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.flat_fee = flat_fee
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode
        
        # ==============================================
        # ACTION SPACE
        # ==============================================
        self.discrete_actions = discrete_actions
        
        if discrete_actions:
            # Discrete acties: 0=Hold, 1=Buy, 2=Sell
            self.action_space = spaces.Discrete(3)
        else:
            # Continue actie: -1 (sell) tot +1 (buy)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(1,), 
                dtype=np.float32
            )
        
        # ==============================================
        # OBSERVATION SPACE
        # ==============================================
        # Observation bevat:
        # 1. Historische features (window_size x num_features)
        # 2. Portfolio state: [balance_ratio, btc_ratio, unrealized_pnl]
        
        # Features zijn genormaliseerd, dus bounds zijn ruim
        self.observation_space = spaces.Dict({
            'features': spaces.Box(
                low=-np.inf, 
                high=np.inf,
                shape=(window_size, self.num_features),
                dtype=np.float32
            ),
            'portfolio': spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(4,),  # balance, btc_held, avg_buy_price, unrealized_pnl
                dtype=np.float32
            )
        })
        
        # Als we een flat observation willen (voor MLP)
        # kunnen we ook een simpele Box gebruiken
        self._flat_obs_size = self.window_size * self.num_features + 4
        
        # Bepaal totaal aantal stappen
        self._total_steps = len(self.prices)
        
        # State variabelen
        self.reset()
        
        logger.info(f"Environment geïnitialiseerd: {self._total_steps} timesteps, {self.num_features} features")
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset de environment naar begin state.
        
        Args:
            seed: Random seed voor reproducibility
            options: Extra opties (niet gebruikt)
            
        Returns:
            observation: Initiële observatie
            info: Extra informatie
        """
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.balance = self.initial_balance  # Cash in USDT
        self.btc_held = 0.0  # Hoeveelheid BTC
        self.avg_buy_price = 0.0  # Gemiddelde aankoopprijs
        
        # Reset tracking variabelen
        logger.debug(f"Environment reset - Initial balance: ${self.initial_balance:.2f}")
        
        # Bepaal startpositie (random of vast)
        if self.random_start:
            # Random start within first X% of data for better generalization
            if self._use_sequences or self._use_raw_features:
                max_start = max(1, int(len(self.prices) * self.random_start_range))
                self.current_step = self.np_random.integers(0, max_start)
            else:
                max_start = max(self.window_size + 1, int(len(self.prices) * self.random_start_range))
                self.current_step = self.np_random.integers(self.window_size, max_start)
            logger.debug(f"Random start at step {self.current_step}")
        else:
            # Vaste startpositie
            if self._use_sequences or self._use_raw_features:
                self.current_step = 0
            else:
                self.current_step = self.window_size
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_portfolio_value = self.initial_balance
        self.trade_history = []
        self.portfolio_values = []
        self._episode_step = 0
        
        # Bereken initiële observatie
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, 
        action: Any
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Voer één trading stap uit.
        
        Args:
            action: Actie van de agent (0=hold, 1=buy, 2=sell of continue)
            
        Returns:
            observation: Nieuwe observatie na actie
            reward: Reward voor deze stap
            terminated: True als episode klaar is
            truncated: True als episode afgekapt is
            info: Extra informatie
        """
        # Huidige prijs
        current_price = self.prices[self.current_step]
        
        # Vorige portfolio waarde voor reward berekening
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # ==============================================
        # VOER ACTIE UIT
        # ==============================================
        trade_info = self._execute_action(action, current_price)
        
        # Ga naar volgende tijdstap
        self.current_step += 1
        self._episode_step += 1
        
        # Nieuwe prijs na actie
        new_price = self.prices[self.current_step] if self.current_step < len(self.prices) else current_price
        
        # ==============================================
        # BEREKEN REWARD
        # ==============================================
        reward = self._calculate_reward(prev_portfolio_value, trade_info)
        
        # ==============================================
        # CHECK EPISODE EINDE
        # ==============================================
        # Episode eindigt als:
        # 1. We aan het einde van de data zijn
        # 2. We failliet zijn (portfolio < 10% van start)
        terminated = False
        truncated = False
        
        current_value = self._calculate_portfolio_value()
        
        if self.current_step >= len(self.prices) - 1:
            truncated = True  # Einde van data
        
        # Episode length limit
        if self.max_episode_steps > 0 and self._episode_step >= self.max_episode_steps:
            truncated = True
            
        if current_value < self.initial_balance * 0.1:
            # Failliet: reset balance, grote penalty, maar episode gaat door
            reward -= 10  # Grote penalty voor faillissement
            logger.warning(
                f"BANKRUPTCY @ step {self.current_step}: "
                f"Portfolio value ${current_value:.2f} < 10% of initial ${self.initial_balance:.2f}. "
                f"Resetting balance to ${self.initial_balance:.2f}. "
                f"Total trades: {self.total_trades}, Win rate: {self.winning_trades}/{self.total_trades}"
            )
            # Reset: verkoop alles en herstel startkapitaal
            self.btc_held = 0.0
            self.avg_buy_price = 0.0
            self.balance = self.initial_balance
            current_value = self.initial_balance
        
        # ==============================================
        # EPISODE EINDE: Force close open posities
        # ==============================================
        # Als de episode eindigt met een open positie, sluit die af
        # zodat PnL correct berekend wordt (niet als "verlies" tellen)
        if (terminated or truncated) and self.btc_held > 0:
            close_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            sell_value = self.btc_held * close_price
            fee = sell_value * self.transaction_fee + self.flat_fee
            net_value = sell_value - fee
            cost_basis = self.btc_held * self.avg_buy_price
            profit = net_value - cost_basis
            
            self.balance += net_value
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            self.total_profit += profit
            self.btc_held = 0.0
            self.avg_buy_price = 0.0
            
            # Herbereken portfolio na force close
            current_value = self._calculate_portfolio_value()
        
        # Track portfolio values
        self.portfolio_values.append(current_value)
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        
        # =====================================
        # LOGGING
        # =====================================
        # Log trade details voor debugging
        if _LOG_TRADE_DETAILS and trade_info['executed']:
            logger.info(
                f"Trade @ step {self.current_step}: {trade_info['type'].upper()} | "
                f"Amount: {trade_info['amount']:.6f} BTC | "
                f"Price: ${trade_info['price']:.2f} | "
                f"Fee: ${trade_info['fee']:.4f} | "
                f"Portfolio: ${current_value:.2f}"
            )
        
        # Log periodic step info voor performance tracking
        if self.current_step % _LOG_STEP_INTERVAL == 0:
            logger.debug(
                f"Step {self.current_step}: "
                f"Balance=${self.balance:.2f}, "
                f"BTC={self.btc_held:.6f}, "
                f"Value=${current_value:.2f}, "
                f"Return={((current_value/self.initial_balance)-1)*100:.2f}%"
            )
        
        # Bereken drawdown voor logging
        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # Log warnings voor kritieke situaties (max elke 1000 steps om spam te voorkomen)
        if current_drawdown > 0.2 and self.current_step % 1000 == 0:
            logger.warning(
                f"High drawdown alert @ step {self.current_step}: "
                f"{current_drawdown*100:.1f}% drawdown from peak ${self.max_portfolio_value:.2f}"
            )
        
        # Maak observatie en info
        obs = self._get_observation()
        info = self._get_info()
        info['trade_info'] = trade_info
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(
        self, 
        action: Any, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        Voer de gegeven actie uit.
        
        Args:
            action: Actie (discrete of continue)
            current_price: Huidige BTC prijs
            
        Returns:
            Dictionary met trade informatie
        """
        trade_info = {
            'action': action,
            'executed': False,
            'type': 'hold',
            'amount': 0.0,
            'price': current_price,
            'fee': 0.0
        }
        
        # Converteer actie naar trade
        if not self.discrete_actions:
            action_value = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
            # Clamp to [-1, 1]
            action_value = max(-1.0, min(1.0, action_value))
            # Continue actie: magnitude = fractie, teken = richting
            # action > 0 = buy met fractie |action| van balance
            # action < 0 = sell met fractie |action| van held BTC
            trade_fraction = abs(action_value)
        else:
            discrete_action = int(action)
            action_value = 0.0
            trade_fraction = 1.0

        # ==============================================
        # BUY ORDER (action > 0 of discrete_action == 1)
        # ==============================================
        is_buy = (not self.discrete_actions and action_value > 0) or \
                 (self.discrete_actions and discrete_action == 1)
        is_sell = (not self.discrete_actions and action_value < 0) or \
                  (self.discrete_actions and discrete_action == 2)

        if is_buy:
            # Koop met fractie van beschikbare balance
            max_buy_value = self.balance * self.max_position * trade_fraction

            if max_buy_value > 1.0 + self.flat_fee:  # Minimale trade grootte $1 + flat fee
                # Bereken hoeveelheid BTC (minus percentage fee en flat fee)
                fee = max_buy_value * self.transaction_fee + self.flat_fee
                buy_value = max_buy_value - fee
                btc_amount = buy_value / current_price

                # Update gemiddelde aankoopprijs (weighted average)
                total_btc = self.btc_held + btc_amount
                if total_btc > 0:
                    self.avg_buy_price = (
                        (self.avg_buy_price * self.btc_held + current_price * btc_amount)
                        / total_btc
                    )

                # Update balans
                self.balance -= max_buy_value
                self.btc_held += btc_amount

                # Update trade info
                trade_info['executed'] = True
                trade_info['type'] = 'buy'
                trade_info['amount'] = btc_amount
                trade_info['fee'] = fee

                self.total_trades += 1
                self.trade_history.append(trade_info.copy())

        # ==============================================
        # SELL ORDER (action < 0 of discrete_action == 2)
        # ==============================================
        elif is_sell:
            # Verkoop fractie van held BTC
            sell_btc = self.btc_held * trade_fraction

            if sell_btc > 0 and sell_btc * current_price > 1.0 + self.flat_fee:  # Minimale trade grootte $1 + flat fee
                sell_value = sell_btc * current_price
                fee = sell_value * self.transaction_fee + self.flat_fee
                net_value = sell_value - fee

                # Bereken profit/loss voor deze trade
                cost_basis = sell_btc * self.avg_buy_price
                profit = net_value - cost_basis

                # Update balans
                self.balance += net_value

                # Update trade info
                trade_info['executed'] = True
                trade_info['type'] = 'sell'
                trade_info['amount'] = sell_btc
                trade_info['fee'] = fee
                trade_info['profit'] = profit

                # Track winning trades
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                self.total_profit += profit

                # Update positie
                self.btc_held -= sell_btc
                if self.btc_held < 1e-10:
                    self.btc_held = 0.0
                    self.avg_buy_price = 0.0

                self.trade_history.append(trade_info.copy())
        
        return trade_info
    
    def _calculate_reward(
        self, 
        prev_portfolio_value: float, 
        trade_info: Dict[str, Any]
    ) -> float:
        """
        Bereken de reward voor deze stap.
        
        Reward design principes:
        - Genormaliseerde rewards in bereik ~[-1, 1] voor stabiel PPO leren
        - Portfolio return als basis signaal (fees zitten er al in)
        - Kleine bonus voor winstgevende sells, geen idle penalty
        - Drawdown penalty alleen bij grote drawdowns (>15%)
        - Reward clipping om extreme waarden te voorkomen
        
        Args:
            prev_portfolio_value: Portfolio waarde voor de actie
            trade_info: Informatie over uitgevoerde trade
            
        Returns:
            Reward waarde (float), geclipt naar [-10, 10]
        """
        current_value = self._calculate_portfolio_value()
        
        # =====================================
        # BASIS REWARD: Portfolio return (genormaliseerd)
        # =====================================
        # Simpele percentage verandering, NIET vermenigvuldigd met 100
        # Dit houdt rewards in een klein bereik dat PPO goed kan leren
        pnl_pct = (current_value - prev_portfolio_value) / prev_portfolio_value
        reward = pnl_pct * self.reward_scaling
        
        # =====================================
        # TRADE SIGNALEN (klein en proportioneel)
        # =====================================
        if trade_info['executed'] and trade_info['type'] == 'sell':
            profit = trade_info.get('profit', 0.0)
            profit_pct = profit / self.initial_balance
            if profit > 0:
                # Bonus proportioneel aan winst
                reward += profit_pct * 2.0
            else:
                # Penalty proportioneel aan verlies
                reward += profit_pct * 1.0  # profit_pct is al negatief
            # Kleine bonus voor het VOLTOOIEN van een round-trip trade
            # Motiveert de agent om daadwerkelijk buy→sell cycles te doen
            reward += 0.005
        
        # =====================================
        # HOLD MET POSITIE: Unrealized PnL tracking
        # =====================================
        if self.btc_held > 0 and not trade_info['executed']:
            current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            unrealized_pnl = (current_price - self.avg_buy_price) / self.avg_buy_price
            if unrealized_pnl > 0:
                # Kleine beloning voor vasthouden van winnende positie
                reward += unrealized_pnl * 0.1
            elif unrealized_pnl < -0.02:
                # Lichte druk om verliezende posities te sluiten (>2% verlies)
                reward += unrealized_pnl * 0.05  # negatief, proportioneel
        
        # GEEN idle penalty - laat de agent leren wanneer NIET traden de beste actie is
        
        # =====================================
        # RISK-ADJUSTED: Drawdown penalty
        # =====================================
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if drawdown > 0.15:  # > 15% drawdown
            reward -= (drawdown - 0.15) * 0.5
        
        # =====================================
        # REWARD CLIPPING: voorkom extreme waarden
        # =====================================
        reward = np.clip(reward, -10.0, 10.0)
        
        return reward
    
    def _calculate_portfolio_value(self) -> float:
        """
        Bereken de totale portfolio waarde.
        
        Portfolio waarde = Cash + (BTC * huidige prijs)
        
        Returns:
            Totale waarde in USDT
        """
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return self.balance + (self.btc_held * current_price)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Maak de huidige observatie.
        
        Returns:
            Dictionary met 'features' en 'portfolio' arrays
        """
        if self._use_raw_features:
            # Raw features mode: create sequence on-the-fly (numpy view, zero-copy)
            idx = min(self.current_step, len(self._raw_features) - self.window_size)
            feature_window = self._raw_features[idx:idx + self.window_size]
        elif self._use_sequences:
            # Sequences mode: direct de sequence op current_step pakken
            idx = min(self.current_step, len(self.sequences) - 1)
            feature_window = self.sequences[idx]
        else:
            # DataFrame mode: window van features maken
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            
            # Haal features op en pad indien nodig
            feature_window = self.features[start_idx:end_idx]
            
            if len(feature_window) < self.window_size:
                # Pad met eerste rij als we niet genoeg historie hebben
                padding = np.tile(
                    feature_window[0], 
                    (self.window_size - len(feature_window), 1)
                )
                feature_window = np.vstack([padding, feature_window])
        
        # Portfolio state
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        portfolio_value = self._calculate_portfolio_value()
        
        # Unrealized PnL
        if self.btc_held > 0 and self.avg_buy_price > 0:
            unrealized_pnl = (current_price - self.avg_buy_price) / self.avg_buy_price
        else:
            unrealized_pnl = 0.0
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Genormaliseerde cash
            self.btc_held * current_price / portfolio_value if portfolio_value > 0 else 0,  # BTC ratio
            unrealized_pnl,  # Unrealized PnL ratio
            portfolio_value / self.initial_balance  # Portfolio ratio
        ], dtype=np.float32)
        
        return {
            'features': feature_window.astype(np.float32),
            'portfolio': portfolio_state
        }
    
    def get_flat_observation(self) -> np.ndarray:
        """
        Krijg een platte observatie vector (voor MLP modellen).
        
        Returns:
            1D numpy array met alle features + portfolio state
        """
        obs = self._get_observation()
        features_flat = obs['features'].flatten()
        return np.concatenate([features_flat, obs['portfolio']])
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Maak info dictionary met nuttige metrics.
        
        Returns:
            Dictionary met trading metrics
        """
        portfolio_value = self._calculate_portfolio_value()
        
        # Bereken Sharpe ratio als we genoeg data hebben
        sharpe_ratio = 0.0
        if len(self.portfolio_values) > 10:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Per-second data
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'total_profit': self.total_profit,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        }
    
    def render(self):
        """
        Render de huidige state (print naar console).
        """
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"\n=== Step {info['step']} ===")
            print(f"Portfolio: ${info['portfolio_value']:.2f}")
            print(f"Balance: ${info['balance']:.2f}")
            print(f"BTC: {info['btc_held']:.6f}")
            print(f"Total Return: {info['total_return']*100:.2f}%")
            print(f"Trades: {info['total_trades']} (Win rate: {info['win_rate']*100:.1f}%)")


class FlatCryptoTradingEnv(CryptoTradingEnv):
    """
    Versie van CryptoTradingEnv met flat observation space.
    
    Dit is geschikt voor MLP-gebaseerde modellen die geen 2D input nodig hebben.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override observation space naar flat Box
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._flat_obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self.get_flat_observation(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self.get_flat_observation(), reward, terminated, truncated, info


if __name__ == "__main__":
    # Test de environment
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    # Maak dummy data voor testen
    n_samples = 1000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='s'),
        'open': 40000 + np.random.randn(n_samples).cumsum() * 10,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.rand(n_samples) * 100,
        'vwap': 0,
    })
    df['close'] = df['open'] + np.random.randn(n_samples) * 5
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n_samples)) * 2
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n_samples)) * 2
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    
    # Test discrete environment
    env = CryptoTradingEnv(
        df=df,
        feature_columns=feature_cols,
        discrete_actions=True,
        render_mode='human'
    )
    
    obs, info = env.reset()
    print(f"Observation shape: features={obs['features'].shape}, portfolio={obs['portfolio'].shape}")
    
    # Random agent test
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"\n=== Test Complete ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final return: {info['total_return']*100:.2f}%")
