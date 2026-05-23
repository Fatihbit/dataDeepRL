"""
Data Loader Module voor BTC L2 Order Book Data
==============================================

Dit module laadt en verwerkt de Binance L2 (Level 2) order book data
voor gebruik in reinforcement learning modellen.

De data bevat per-seconde aggregaties van:
- OHLCV (Open, High, Low, Close, Volume)
- Bid/Ask prices en volumes
- Order flow metrics (spread, imbalance)

Auteur: DataDeepRL Team
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# =====================================
# LOGGING SETUP
# =====================================
# Logger voor DataLoader module
# Gebruik logging.DEBUG voor gedetailleerde data processing info
logger = logging.getLogger(__name__)

# Feature engineering constanten
# Deze waarden zijn gebaseerd op best practices voor financial time series
RETURN_PERIODS = [5, 10, 30, 60]        # Periodes voor return berekening (in tijdstappen)
SMA_WINDOWS = [10, 30, 60]               # Rolling windows voor Simple Moving Average
EMA_SPANS = [10, 30]                     # Spans voor Exponential Moving Average
VOLATILITY_WINDOWS = [10, 30, 60]        # Windows voor volatiliteitsberekening
RSI_PERIOD = 14                          # Standaard RSI periode
MACD_FAST = 12                           # MACD fast EMA
MACD_SLOW = 26                           # MACD slow EMA
MACD_SIGNAL = 9                          # MACD signal line


class BTCDataLoader:
    """
    DataLoader voor BTC L2 order book data.
    
    Deze class laadt parquet bestanden, combineert ze, en bereidt
    de data voor als input voor DeepLOB of MLP modellen.
    
    Attributen:
        data_dir (str): Directory met parquet bestanden
        window_size (int): Aantal tijdstappen voor lookback window
        scaler: Scaler voor feature normalisatie
        
    Voorbeeld:
        >>> loader = BTCDataLoader(data_dir='btc_l2_data', window_size=100)
        >>> train_data, val_data, test_data = loader.load_and_split()
    """
    
    def __init__(
        self, 
        data_dir: str = 'btc_l2_data',
        window_size: int = 100,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
        scaler_type: str = 'standard'
    ):
        """
        Initialiseer de DataLoader.
        
        Args:
            data_dir: Pad naar directory met parquet bestanden
            window_size: Aantal tijdstappen voor lookback window (voor CNN/LSTM)
            train_ratio: Fractie van data voor training
            val_ratio: Fractie van data voor validatie (rest = test)
            normalize: Of features genormaliseerd moeten worden
            scaler_type: Type scaler ('standard' of 'minmax')
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        
        # Selecteer scaler type
        # StandardScaler: (x - mean) / std -> waarden rond 0
        # MinMaxScaler: (x - min) / (max - min) -> waarden tussen 0 en 1
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        self.df = None  # Geladen DataFrame
        self.feature_names = None  # Namen van features
        
        logger.info(f"DataLoader geïnitialiseerd met data_dir={data_dir}, window_size={window_size}")
    
    def load_data(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Laad alle parquet bestanden uit de data directory.
        
        Args:
            max_files: Maximum aantal bestanden om te laden (None = alles)
            
        Returns:
            DataFrame met alle data gecombineerd
            
        Raises:
            FileNotFoundError: Als geen parquet bestanden gevonden
        """
        # Zoek alle parquet bestanden (exclusief combined files)
        pattern = os.path.join(self.data_dir, 'BTCUSDT_*.parquet')
        files = sorted(glob.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"Geen parquet bestanden gevonden in {self.data_dir}")
        
        # Limiteer aantal bestanden indien gewenst
        if max_files:
            files = files[:max_files]
            
        logger.info(f"Laden van {len(files)} parquet bestanden...")
        
        # Laad en combineer alle bestanden
        dfs = []
        for i, file in enumerate(files):
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
                
                # Log voortgang elke 100 bestanden
                if (i + 1) % 100 == 0:
                    logger.info(f"  Geladen: {i + 1}/{len(files)} bestanden")
                    
            except Exception as e:
                logger.warning(f"Fout bij laden {file}: {e}")
                
        # Combineer alle DataFrames
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Sorteer op timestamp
        if 'timestamp' in self.df.columns:
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data geladen: {len(self.df)} rijen, {len(self.df.columns)} kolommen")
        
        return self.df
    
    def create_features(self) -> pd.DataFrame:
        """
        Maak extra features aan voor het RL model.
        
        Dit voegt technische indicatoren en afgeleide features toe:
        - Returns (prijsveranderingen)
        - Moving averages
        - Volatiliteit
        - Order flow imbalance
        
        Returns:
            DataFrame met extra features
        """
        if self.df is None:
            raise ValueError("Eerst load_data() aanroepen!")
            
        df = self.df.copy()
        initial_rows = len(df)
        logger.info(f"Starting feature engineering on {initial_rows:,} rows...")
        
        # ========================================
        # PRIJSRETURNS
        # ========================================
        # Log returns zijn beter voor ML dan absolute returns omdat:
        # 1. Ze zijn additief over tijd (sum of log returns = total log return)
        # 2. Ze zijn beter genormaliseerd (relatieve verandering)
        # 3. Ze zijn symmetrisch voor up/down moves
        # Formule: log(price_t / price_t-1) ≈ (price_t - price_t-1) / price_t-1 voor kleine changes
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        logger.debug("Computed log returns")
        
        # Cumulatieve returns over verschillende periodes
        # Dit geeft de agent context over recente trends
        for period in RETURN_PERIODS:
            df[f'return_{period}s'] = df['close'].pct_change(period)
        logger.debug(f"Computed returns for periods: {RETURN_PERIODS}")
        
        # ========================================
        # MOVING AVERAGES
        # ========================================
        # Simple Moving Average (SMA) - gemiddelde prijs over N periodes
        # Nut: identificeert trend richting, support/resistance levels
        # Prijs > SMA = bullish, Prijs < SMA = bearish
        for window in SMA_WINDOWS:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            
            # Ratio van prijs tot SMA (mean reversion indicator)
            # > 1 = prijs boven gemiddelde (mogelijk overbought)
            # < 1 = prijs onder gemiddelde (mogelijk oversold)
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        logger.debug(f"Computed SMAs for windows: {SMA_WINDOWS}")
        
        # Exponential Moving Average (EMA) - recente prijzen wegen zwaarder
        # Reageert sneller op recente prijsveranderingen dan SMA
        # Formule: EMA_t = α * price_t + (1-α) * EMA_{t-1}, waar α = 2/(span+1)
        for span in EMA_SPANS:
            df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
        logger.debug(f"Computed EMAs for spans: {EMA_SPANS}")
        
        # ========================================
        # VOLATILITEIT
        # ========================================
        # Rolling standard deviation van returns
        # Hogere volatiliteit = meer risico maar ook meer opportunity
        # Dit helpt de agent het risico te schatten
        for window in VOLATILITY_WINDOWS:
            df[f'volatility_{window}'] = df['log_return'].rolling(window).std()
        logger.debug(f"Computed volatility for windows: {VOLATILITY_WINDOWS}")
        
        # True Range - maat voor volledige intraday price range
        # TR = max(high - low, |high - prev_close|, |low - prev_close|)
        # Dit vangt gaps en extended ranges die niet in OHLC zitten
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']                # Intraday range
        tr2 = abs(df['high'] - prev_close)          # Gap up gevangen
        tr3 = abs(df['low'] - prev_close)           # Gap down gevangen
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range (ATR) - gemiddelde volatiliteit over N periodes
        # Vaak gebruikt voor stop-loss berekening en position sizing
        df['atr_14'] = df['true_range'].rolling(RSI_PERIOD).mean()
        logger.debug("Computed True Range and ATR")
        
        # ========================================
        # VOLUME FEATURES
        # ========================================
        # Volume moving average - baseline voor normale volume
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        # Volume ratio - huidig volume vs normaal
        # > 1 = hoger dan normaal volume (vaak bij belangrijke price moves)
        # < 1 = lager dan normaal volume (consolidatie)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # VWAP difference - Volume Weighted Average Price afwijking
        # Positief = prijs boven VWAP (bullish)
        # Negatief = prijs onder VWAP (bearish)
        # Institutionele traders gebruiken VWAP als benchmark
        df['vwap_diff'] = df['close'] - df['vwap']
        logger.debug("Computed volume features")
        
        # ========================================
        # ORDER FLOW / MARKET MICROSTRUCTURE
        # ========================================
        # Dit zijn de meest waardevolle features voor market making/trading
        # Ze geven inzicht in supply/demand dynamics die niet in price zitten
        
        # Bid-Ask spread als percentage
        # Hogere spread = minder liquiditeit, hogere trading kosten
        # Lagere spread = meer liquiditeit, efficiëntere markt
        df['spread_pct_clean'] = (df['ask_price'] - df['bid_price']) / df['vwap'] * 100
        
        # Order Flow Imbalance (OFI)
        # Meet de verhouding tussen ask en bid volume
        # Positief = meer verkoop orders (bearish druk)
        # Negatief = meer koop orders (bullish druk)
        # Range: -1 tot +1
        df['order_imbalance'] = (df['ask_volume'] - df['bid_volume']) / (df['ask_volume'] + df['bid_volume'])
        
        # Trade count imbalance - wie is actiever?
        # Vergelijkbaar met OFI maar dan op basis van aantal trades
        # +1 epsilon om division by zero te voorkomen
        df['trade_imbalance'] = (df['ask_trades'] - df['bid_trades']) / (df['ask_trades'] + df['bid_trades'] + 1)
        logger.debug("Computed order flow features")
        
        # ========================================
        # MOMENTUM INDICATOREN
        # ========================================
        # Relative Strength Index (RSI)
        # RSI meet of een asset overbought (>70) of oversold (<30) is
        # Berekening:
        # 1. Bereken up moves en down moves
        # 2. Neem rolling average van beide
        # 3. RS = avg_gain / avg_loss
        # 4. RSI = 100 - (100 / (1 + RS))
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / (loss + 1e-10)  # Kleine epsilon om deling door 0 te voorkomen
        df['rsi_14'] = 100 - (100 / (1 + rs))
        logger.debug("Computed RSI")
        
        # MACD (Moving Average Convergence Divergence)
        # MACD meet momentum door verschil tussen korte en lange EMA
        # MACD Line = EMA(12) - EMA(26)
        # Signal Line = EMA(9) van MACD Line
        # Histogram = MACD - Signal (positief = bullish momentum)
        ema_12 = df['close'].ewm(span=MACD_FAST).mean()
        ema_26 = df['close'].ewm(span=MACD_SLOW).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        logger.debug("Computed MACD")
        
        # ========================================
        # OPSCHONEN
        # ========================================
        # Verwijder rijen met NaN (door rolling windows)
        # De eerste window_size rijen hebben incomplete features
        df = df.dropna().reset_index(drop=True)
        rows_removed = initial_rows - len(df)
        
        # Bewaar feature namen voor later
        self.feature_names = df.columns.tolist()
        
        self.df = df
        
        # Log samenvatting
        logger.info(
            f"Feature engineering complete: "
            f"{len(df.columns)} features, "
            f"{len(df):,} rows "
            f"({rows_removed:,} rows removed due to NaN from rolling windows)"
        )
        
        # Log feature statistieken voor debugging
        logger.debug(f"Feature columns: {self.feature_names}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Retourneer de kolommen die als features gebruikt worden voor het model.
        
        Returns:
            Lijst met feature kolomnamen
        """
        # Selecteer numerieke kolommen die als features dienen
        # Exclusief timestamp en andere niet-feature kolommen
        exclude_cols = ['timestamp', 'datetime']
        
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(self.df[col])]
        
        return feature_cols
    
    def prepare_sequences(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        price_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bereid sequences voor voor het DeepLOB model (CNN + LSTM).
        
        Dit creëert sliding windows van features met de bijbehorende
        toekomstige prijsverandering als target.
        
        Args:
            df: DataFrame met features
            feature_cols: Lijst met feature kolommen (None = auto detect)
            price_col: Kolom voor prijs (voor target berekening)
            
        Returns:
            Tuple van (X, y, prices):
            - X: shape (samples, window_size, num_features)
            - y: shape (samples,) - toekomstige return
            - prices: shape (samples,) - prijzen voor elke sequence
        """
        # Auto-detect feature columns als niet gegeven
        if feature_cols is None:
            feature_cols = self.get_feature_columns()
        
        # Haal features op als numpy array
        features = df[feature_cols].values
        prices = df[price_col].values
        
        # Bereken toekomstige returns (1 stap vooruit)
        # Dit wordt de target die het model moet voorspellen
        future_returns = np.zeros(len(prices))
        future_returns[:-1] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        X_list = []
        y_list = []
        price_list = []
        
        # Maak sliding windows
        for i in range(self.window_size, len(features)):
            # Feature window: laatste window_size tijdstappen
            X_list.append(features[i - self.window_size:i])
            # Target: return van huidige naar volgende tijdstap
            y_list.append(future_returns[i])
            # Prijs bij einde van window (voor trading)
            price_list.append(prices[i])
        
        X = np.array(X_list)
        y = np.array(y_list)
        prices_out = np.array(price_list)
        
        logger.info(f"Sequences gemaakt: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y, prices_out
    
    def normalize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray = None, 
        X_test: np.ndarray = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Normaliseer features met fitted scaler.
        
        BELANGRIJK: Fit alleen op training data om data leakage te voorkomen!
        
        Args:
            X_train: Training features met shape (samples, window, features)
            X_val: Validatie features (optioneel)
            X_test: Test features (optioneel)
            
        Returns:
            Genormaliseerde arrays
        """
        # Reshape naar 2D voor scaler: (samples * window, features)
        n_samples, window, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        
        # Fit scaler ALLEEN op training data
        self.scaler.fit(X_train_2d)
        
        # Transform training data
        X_train_normalized = self.scaler.transform(X_train_2d).reshape(n_samples, window, n_features)
        
        result = [X_train_normalized]
        
        # Transform validation en test data met dezelfde scaler
        if X_val is not None:
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_normalized = self.scaler.transform(X_val_2d).reshape(X_val.shape)
            result.append(X_val_normalized)
            
        if X_test is not None:
            X_test_2d = X_test.reshape(-1, n_features)
            X_test_normalized = self.scaler.transform(X_test_2d).reshape(X_test.shape)
            result.append(X_test_normalized)
        
        logger.info("Features genormaliseerd")
        
        return tuple(result)
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        prices: np.ndarray = None,
        train_ratio: float = None,
        val_ratio: float = None
    ) -> Tuple:
        """
        Split data in train/validation/test sets.
        
        BELANGRIJK: We splitsen chronologisch (niet random) omdat
        dit tijdreeksdata is. Random splitten zou toekomstige info lekken.
        
        Args:
            X: Feature arrays
            y: Target arrays
            prices: Prijzen arrays (optioneel)
            train_ratio: Override train ratio
            val_ratio: Override val ratio
            
        Returns:
            Als prices is None: X_train, X_val, X_test, y_train, y_val, y_test
            Als prices gegeven: (train_data, val_data, test_data) waar elke tuple (X, y, prices) bevat
        """
        n_samples = len(X)
        
        # Gebruik override of default ratios
        tr = train_ratio if train_ratio is not None else self.train_ratio
        vr = val_ratio if val_ratio is not None else self.val_ratio
        
        # Bereken split indices
        train_end = int(n_samples * tr)
        val_end = int(n_samples * (tr + vr))
        
        # Split chronologisch
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        logger.info(f"Data gesplitst: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Return format hangt af van of prices gegeven is
        if prices is not None:
            prices_train = prices[:train_end]
            prices_val = prices[train_end:val_end]
            prices_test = prices[val_end:]
            return (
                (X_train, y_train, prices_train),
                (X_val, y_val, prices_val),
                (X_test, y_test, prices_test)
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_and_prepare(
        self, 
        max_files: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: laad, verwerk en split data.
        
        Dit is de hoofd-functie om data klaar te maken voor training.
        
        Args:
            max_files: Maximum aantal bestanden om te laden
            
        Returns:
            Dictionary met alle data arrays
        """
        # Stap 1: Laad data
        self.load_data(max_files=max_files)
        
        # Stap 2: Maak features
        self.create_features()
        
        # Stap 3: Selecteer feature kolommen
        feature_cols = self.get_feature_columns()
        logger.info(f"Geselecteerde features: {len(feature_cols)}")
        
        # Stap 4: Maak sequences voor DeepLOB
        X, y = self.prepare_sequences(self.df, feature_cols)
        
        # Stap 5: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Stap 6: Normaliseer als gewenst
        if self.normalize:
            X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'scaler': self.scaler
        }


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standalone functie om features aan te maken.
    
    Handig als je al een DataFrame hebt en alleen features wilt toevoegen.
    
    Args:
        df: DataFrame met OHLCV data
        
    Returns:
        DataFrame met extra features
    """
    loader = BTCDataLoader()
    loader.df = df.copy()
    return loader.create_features()


if __name__ == "__main__":
    # Test de DataLoader
    logging.basicConfig(level=logging.INFO)
    
    loader = BTCDataLoader(
        data_dir='btc_l2_data',
        window_size=100
    )
    
    # Test met beperkt aantal bestanden voor snelheid
    data = loader.load_and_prepare(max_files=10)
    
    print("\n=== Data Samenvatting ===")
    print(f"Training samples: {data['X_train'].shape}")
    print(f"Validation samples: {data['X_val'].shape}")
    print(f"Test samples: {data['X_test'].shape}")
    print(f"Aantal features: {len(data['feature_cols'])}")
