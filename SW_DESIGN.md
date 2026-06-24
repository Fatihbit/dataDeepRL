# Software Design Document — DataDeepRL

## 1. Inleiding

Dit document beschrijft het gedetailleerde softwareontwerp van DataDeepRL:
klassendiagrammen, toestandsdiagrammen, datastromen en interfacedefinities.
Het bouwt voort op [ARCHITECTURE.md](ARCHITECTURE.md) en gaat dieper in op
de interne structuur van elke component.

---

## 2. Klassendiagram — Volledig systeem

```mermaid
classDiagram
    direction TB

    class CryptoTradingEnv {
        +raw_features: ndarray
        +prices: ndarray
        +window_size: int
        +initial_balance: float
        +transaction_fee: float
        +flat_fee: float
        +discrete_actions: bool
        +balance: float
        +btc_held: float
        +current_step: int
        +reset() obs, info
        +step(action) obs, reward, terminated, truncated, info
        +_get_observation() dict
        +_calculate_portfolio_value() float
        +_calculate_reward() float
    }

    class FlatCryptoTradingEnv {
        +reset() ndarray, info
        +step(action) ndarray, reward, terminated, truncated, info
    }

    class VectorizedTradingEnv {
        +n_envs: int
        +envs: list
        +reset() ndarray
        +step(actions) ndarray, rewards, dones, infos
    }

    class DeepLOB {
        +input_dim: int
        +hidden_dim: int
        +lstm_hidden: int
        +output_dim: int
        +dropout: float
        +conv_block1: ConvBlock
        +conv_block2: ConvBlock
        +inception: InceptionModule
        +lstm: BiLSTM
        +attention: AttentionPooling
        +forward(x) Tensor
    }

    class ConvBlock {
        +conv: Conv1d
        +bn: BatchNorm1d
        +activation: LeakyReLU
        +forward(x) Tensor
    }

    class InceptionModule {
        +branch1: Sequential  ~~1x1~~
        +branch2: Sequential  ~~1x1+3x3~~
        +branch3: Sequential  ~~1x1+5x5~~
        +branch4: Sequential  ~~MaxPool+1x1~~
        +forward(x) Tensor
    }

    class PPOAgent {
        +policy: MLPPolicyNetwork
        +value: MLPValueNetwork
        +rollout_buffer: RolloutBuffer
        +gamma: float
        +clip_range: float
        +n_epochs: int
        +collect_rollout(env) 
        +update() dict
        +save(path)
        +load(path)
    }

    class RolloutBuffer {
        +buffer_size: int
        +observations: ndarray
        +actions: ndarray
        +rewards: ndarray
        +values: ndarray
        +log_probs: ndarray
        +advantages: ndarray
        +add(obs, action, reward, value, log_prob, done)
        +compute_returns_and_advantages(last_value)
        +get(device) dict
        +reset()
    }

    class SACAgent {
        +actor: MLPActor
        +critic1: MLPCritic
        +critic2: MLPCritic
        +critic1_target: MLPCritic
        +critic2_target: MLPCritic
        +replay_buffer: ReplayBuffer
        +alpha: float
        +tau: float
        +update() dict
        +select_action(obs) ndarray
        +save(path)
        +load(path)
    }

    class ReplayBuffer {
        +capacity: int
        +observations: ndarray
        +actions: ndarray
        +rewards: ndarray
        +next_observations: ndarray
        +dones: ndarray
        +ptr: int
        +size: int
        +add(obs, action, reward, next_obs, done)
        +sample(batch_size, device) dict
    }

    class MLPFeatureExtractor {
        +input_dim: int
        +features_dim: int
        +network: Sequential
        +forward(obs) Tensor
    }

    class MLPPolicyNetwork {
        +forward(obs) action, log_prob
    }

    class MLPValueNetwork {
        +forward(obs) value
    }

    CryptoTradingEnv <|-- FlatCryptoTradingEnv
    VectorizedTradingEnv o-- CryptoTradingEnv

    DeepLOB *-- ConvBlock
    DeepLOB *-- InceptionModule

    PPOAgent *-- RolloutBuffer
    PPOAgent *-- MLPPolicyNetwork
    PPOAgent *-- MLPValueNetwork
    PPOAgent o-- DeepLOB

    SACAgent *-- ReplayBuffer
    SACAgent *-- MLPActor
    SACAgent o-- DeepLOB

    MLPPolicyNetwork --|> MLPFeatureExtractor
    MLPValueNetwork --|> MLPFeatureExtractor
```

---

## 3. Klassendiagram — DeepLOB detail

```mermaid
classDiagram
    direction LR

    class DeepLOB {
        +input_dim: int
        +hidden_dim: int = 64
        +lstm_hidden: int = 64
        +num_lstm_layers: int = 2
        +output_dim: int = 64
        +dropout: float = 0.2
        +forward(x: Tensor) Tensor
    }

    class ConvBlock {
        +conv: Conv1d
        +bn: BatchNorm1d
        +activation: LeakyReLU(0.01)
        +forward(x) Tensor
    }

    class InceptionModule {
        +branch1: 1×1 conv
        +branch2: 1×1 → 3×3 conv
        +branch3: 1×1 → 5×5 conv
        +branch4: MaxPool → 1×1 conv
        +forward(x) Tensor  ~~concat 4 branches~~
    }

    class BiLSTM {
        <<nn.LSTM>>
        +bidirectional = True
        +num_layers: int
        +hidden_size: int
        +dropout: float
    }

    class AttentionPooling {
        +attention_weights: Linear
        +forward(x) Tensor  ~~weighted mean~~
    }

    class DeepLOBClassifier {
        +backbone: DeepLOB
        +classifier: Linear  ~~output_dim → 3~~
        +forward(x) logits  ~~voor pretraining~~
    }

    class DeepLOBFeatureExtractor {
        +backbone: DeepLOB
        +forward(x) features  ~~voor RL~~
    }

    DeepLOB *-- ConvBlock : conv_block1, conv_block2
    DeepLOB *-- InceptionModule
    DeepLOB *-- BiLSTM
    DeepLOB *-- AttentionPooling
    DeepLOBClassifier *-- DeepLOB
    DeepLOBFeatureExtractor *-- DeepLOB
```

---

## 4. Toestandsdiagram — Trading Environment

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Running : reset()

    state Running {
        [*] --> Observing
        Observing --> Acting : _get_observation()
        Acting --> Executing : agent geeft actie

        state Executing {
            [*] --> Hold : actie == 0
            [*] --> Buy : actie == 1 AND balance > 0
            [*] --> Sell : actie == 2 AND btc_held > 0
            Hold --> [*]
            Buy --> [*] : update balance, btc_held
            Sell --> [*] : update balance, btc_held
        }

        Executing --> RewardCalc : _calculate_reward()

        state RewardCalc {
            [*] --> PnLDelta : ΔPortfoliowaarde
            PnLDelta --> FeeDeduct : - transactiekosten
            FeeDeduct --> DrawdownCheck
            DrawdownCheck --> Penalty : drawdown > 15%
            DrawdownCheck --> [*] : geen penalty
            Penalty --> [*]
        }

        RewardCalc --> Observing : next step
    }

    Running --> Terminated : data op / bankruptcy
    Running --> Truncated : max_episode_steps bereikt
    Terminated --> [*]
    Truncated --> [*]
```

---

## 5. Toestandsdiagram — PPO trainingsloop

```mermaid
stateDiagram-v2
    [*] --> Initialiseren

    Initialiseren --> Rollout : setup env + buffer

    state Rollout {
        [*] --> StepEnv
        StepEnv --> BufferAdd : obs, action, reward, value, log_prob, done
        BufferAdd --> StepEnv : ptr < buffer_size
        BufferAdd --> GAEBerekenen : buffer vol
        GAEBerekenen --> [*]
    }

    Rollout --> PolicyUpdate

    state PolicyUpdate {
        [*] --> NormAdvantages
        NormAdvantages --> ForwardPass
        ForwardPass --> ClippedLoss : L_CLIP
        ClippedLoss --> ValueLoss : L_VF
        ValueLoss --> EntropyBonus : L_ENT
        EntropyBonus --> Backprop : L = L_CLIP - c1*L_VF + c2*L_ENT
        Backprop --> GradClip
        GradClip --> [*]
    }

    PolicyUpdate --> EvalCheck : update_count % eval_freq == 0

    state EvalCheck {
        [*] --> ValEpisodes
        ValEpisodes --> CompositeScore
        CompositeScore --> SaveBest : score > best_score
        CompositeScore --> [*] : geen verbetering
        SaveBest --> [*]
    }

    EvalCheck --> Rollout : doorgaan
    EvalCheck --> Klaar : total_timesteps bereikt
    Klaar --> [*]
```

---

## 6. Toestandsdiagram — SAC trainingsloop

```mermaid
stateDiagram-v2
    [*] --> Initialiseren

    Initialiseren --> Warmup : vul replay buffer

    state Warmup {
        [*] --> RandomAction
        RandomAction --> BufferStore : (s, a, r, s', done)
        BufferStore --> RandomAction : size < learning_starts
        BufferStore --> [*] : genoeg samples
    }

    Warmup --> Training

    state Training {
        [*] --> StepEnv
        StepEnv --> BufferAdd

        BufferAdd --> SampleBatch : random sample uit ReplayBuffer

        state SampleBatch {
            [*] --> UpdateCritics : Twin Q-network loss
            UpdateCritics --> UpdateActor : maximaliseer Q + entropy
            UpdateActor --> UpdateAlpha : pas temperatuur α aan
            UpdateAlpha --> SoftUpdate : θ_target ← τθ + (1-τ)θ_target
            SoftUpdate --> [*]
        }

        SampleBatch --> EvalCheck
        EvalCheck --> StepEnv : doorgaan
    }

    Training --> Klaar : total_timesteps bereikt
    Klaar --> [*]
```

---

## 7. Datastroom — observatie door het netwerk

```mermaid
flowchart LR
    subgraph ENV["CryptoTradingEnv"]
        RF["raw_features\n(N × 15)"]
        WIN["window slice\n(100 × 15)\nnumpy view"]
        PF["portfolio\n(4,)"]
        RF -->|on-the-fly| WIN
    end

    subgraph OBS["Observatie dict"]
        F["features\n(100 × 15)"]
        P["portfolio\n(4,)"]
        WIN --> F
        PF --> P
    end

    subgraph DEEPLOB["DeepLOB (optioneel)"]
        CB["ConvBlocks\n(100 × hidden_dim)"]
        INC["Inception\n(100 × hidden_dim×4)"]
        LSTM["BiLSTM\n(100 × lstm_hidden×2)"]
        ATT["Attention\n(lstm_hidden×2,)"]
        F --> CB --> INC --> LSTM --> ATT
    end

    subgraph MLP["MLP Netwerk"]
        FLAT["flatten + concat\n(feature_vec + portfolio)"]
        H1["Linear → LayerNorm → LeakyReLU"]
        H2["Linear → LayerNorm → LeakyReLU"]
        OUT["feature vector\n(features_dim,)"]
        ATT --> FLAT
        P --> FLAT
        FLAT --> H1 --> H2 --> OUT
    end

    subgraph HEAD["Output Head"]
        POL["Policy Head\n→ actieverdeling"]
        VAL["Value Head\n→ V(s)"]
        OUT --> POL
        OUT --> VAL
    end
```

---

## 8. Datastroom — rewardberekening

```mermaid
flowchart TD
    A["Vorige portfoliowaarde\nbalance + btc_held × prijs_t-1"]
    B["Huidige portfoliowaarde\nbalance + btc_held × prijs_t"]
    C["ΔPortfolio = B - A"]

    D{Trade\nuitgevoerd?}
    E["- flat_fee (bijv. $1)\n- percentage_fee × waarde"]
    F["Netto reward\n= ΔPortfolio - kosten"]

    G{Drawdown\n> 15%?}
    H["Penalty: -0.1"]
    I["Eindreward"]

    A --> C
    B --> C
    C --> D
    D -->|Ja| E --> F
    D -->|Nee| F
    F --> G
    G -->|Ja| H --> I
    G -->|Nee| I
```

---

## 9. Interface — CryptoTradingEnv

```mermaid
flowchart LR
    subgraph INPUT["Invoer"]
        I1["raw_features: float32\n(N × 15)"]
        I2["prices: float64\n(N,) — denorm. USD"]
        I3["window_size: int = 100"]
        I4["initial_balance: float = 100000"]
        I5["flat_fee: float = 1.0"]
        I6["discrete_actions: bool = True"]
    end

    subgraph ENV["CryptoTradingEnv"]
        R["reset() → obs, info"]
        S["step(action) → obs, reward,\nterminated, truncated, info"]
    end

    subgraph OUTPUT["Uitvoer obs-dict"]
        O1["features: float32\n(window_size × 15)"]
        O2["portfolio: float32\n(4,)\n[balance, btc_held,\navg_buy_price, unrealized_pnl]"]
    end

    subgraph INFO["info-dict (per stap)"]
        N1["portfolio_value: float"]
        N2["total_trades: int"]
        N3["winning_trades: int"]
        N4["total_profit: float"]
        N5["current_price: float"]
    end

    INPUT --> ENV
    ENV --> OUTPUT
    ENV --> INFO
```

---

## 10. Composite Score — berekeningsformule

```mermaid
flowchart TD
    SR["Sharpe Ratio\ngemiddeld rendement / std rendement"]
    RT["Total Return\n(eindwaarde - startwaarde) / startwaarde"]
    DD["Max Drawdown\nmaximale piekdaling in %"]

    C1["clip(Sharpe, -5, 5) / 5\n→ bereik: [-1, 1]"]
    C2["clip(Return, -1, 1)\n→ bereik: [-1, 1]"]
    C3["Drawdown\n→ bereik: [0, 1]"]

    W1["× 0.5\ngewicht rendement"]
    W2["× 0.5\ngewicht risk-adjusted"]
    W3["× 0.2\npenalty drawdown"]

    SUM["Composite Score\n= 0.5×clip(Sharpe)/5\n+ 0.5×clip(Return)\n− 0.2×Drawdown\nbereik: [-1.2, 1.0]"]

    SR --> C1 --> W2 --> SUM
    RT --> C2 --> W1 --> SUM
    DD --> C3 --> W3 --> SUM
```

> Gebruikt voor:
> - **Tijdens training**: checkpoint selectie (`best_model.pt`)
> - **Bij evaluatie**: eindoordeel per episode op testdata

---

## 11. Hyperparameter overzicht

### DeepLOB pretraining

| Parameter | Default | Beschrijving |
|---|---|---|
| `hidden_dim` | 64 | Conv kanalen |
| `lstm_hidden` | 64 | LSTM hidden size |
| `num_lstm_layers` | 2 | Aantal BiLSTM lagen |
| `output_dim` | 64 | Feature vector grootte |
| `dropout` | 0.2 | Regularisatie |
| `lr` | 1e-3 | Learning rate |
| `batch_size` | 512 | Batch grootte |
| `epochs` | 30 | Trainingsepochs |
| `patience` | 1000 | Early stopping |
| `window_size` | 100 | Tijdvenster (tijdstappen) |

### PPO

| Parameter | Default | Beschrijving |
|---|---|---|
| `n_steps` | 2048 | Rollout grootte |
| `n_epochs` | 10 | Update epochs per rollout |
| `batch_size` | 64 | Mini-batch grootte |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clipping (ε) |
| `lr` | 3e-4 | Learning rate |
| `ent_coef` | 0.01 | Entropy bonus coëfficiënt |
| `vf_coef` | 0.5 | Value loss coëfficiënt |
| `max_grad_norm` | 0.5 | Gradient clipping |

### SAC

| Parameter | Default | Beschrijving |
|---|---|---|
| `buffer_size` | 1_000_000 | Replay buffer capaciteit |
| `batch_size` | 256 | Sample batch grootte |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coëfficiënt |
| `alpha` | 0.2 | Initiële entropie temperatuur |
| `lr_actor` | 3e-4 | Actor learning rate |
| `lr_critic` | 3e-4 | Critic learning rate |
| `learning_starts` | 10_000 | Warmup stappen |
| `train_freq` | 1 | Update elke N stappen |

---

## 12. Ontwerppatronen

| Patroon | Toepassing |
|---|---|
| **Strategy** | `PPOAgent` en `SACAgent` zijn uitwisselbaar via dezelfde `train()`-interface |
| **Decorator** | `FlatCryptoTradingEnv` wraps `CryptoTradingEnv` zonder de logica te dupliceren |
| **Template Method** | Alle trainingsscripts volgen dezelfde structuur: args → env → agent → train loop → eval |
| **Circular Buffer** | `ReplayBuffer` overschrijft oudste entries bij volle capaciteit |
| **Factory** | `load_coredata_streaming()` produceert feature/price arrays voor elke split |
| **Frozen Object** | DeepLOB wordt na pretraining bevroren en puur als feature extractor gebruikt |
