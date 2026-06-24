# Testrapport & Testdocumentatie — DataDeepRL

**Gegenereerd:** 2026-06-23 20:30:31

## ✅ ALLE TESTS GESLAAGD

## Samenvatting

| Metric | Waarde |
|---|---|
| Totaal tests | 30 |
| Geslaagd | 30 |
| Gefaald | 0 |
| Errors | 0 |
| Overgeslagen | 0 |
| Slaagpercentage | 100.0% |
| Code-coverage | 29.2% |
| Duur | 3.06s |

## Soorten tests

De testsuite volgt de **testpiramide**: veel snelle unit tests onderaan, aangevuld met non-functionele tests (performance/geheugen) en een laag acceptatietests (end-to-end) bovenaan. Alle tests gebruiken synthetische data en hebben geen echte dataset of getraind model nodig. Per test geeft de kolom **Soort** aan wat voor soort controle het is:

| Soort | Betekenis |
|---|---|
| **Unit test** | Test één component/functie geïsoleerd, zonder externe data of andere modules. |
| **Gedrag** | Controleert of een actie het juiste gevolg heeft (bv. kopen verlaagt de balans). |
| **Contract** | Controleert vorm, datatype of return-signatuur (bv. observatie is (50,15) float32). |
| **Numeriek** | Controleert een exacte waarde of geldig bereik (bv. spread = 1.0, RSI in [0,100]). |
| **Robuustheid** | Controleert numerieke gezondheid (geen NaN, determinisme, gradiënten stromen). |
| **Performance** | Non-functioneel: snelheid moet boven een drempel liggen (steps/s, latency). |
| **Geheugen** | Non-functioneel: data wordt zuinig opgeslagen (zero-copy sliding window). |
| **Acceptatie** | End-to-end: een compleet gebruikersscenario werkt van begin tot eind. |

## Geteste onderdelen

| Module | Soort | Onderwerp | Tests |
|---|---|---|---|
| `test_acceptance` | Acceptatietest (E2E) | Acceptatie — end-to-end gebruikersscenario's (Given-When-Then) | 4 |
| `test_deeplob` | Unit test (component/smoke) | DeepLOB-model — outputvorm, NaN-check, determinisme, gradiënten | 4 |
| `test_features` | Unit test | Feature engineering — spread, order imbalance, RSI, returns, momentum | 6 |
| `test_performance` | Non-functionele test | Non-functioneel — throughput, latency, geheugen-efficiëntie | 5 |
| `test_trading_env` | Unit test | CryptoTradingEnv — reset, step, kopen/verkopen, portfolio, observaties | 11 |

## Wat is er getest

### `test_acceptance`

_Acceptatie — end-to-end gebruikersscenario's (Given-When-Then)_

**Soort:** Acceptatietest (E2E)

Test of complete scenario's vanuit gebruikersperspectief werken: een hele episode spelen, geldige rapport-metrics, buy-and-hold winst.

| Test | Wat wordt getest | Soort | Uitslag | Tijd |
|---|---|---|---|---|
| `test_buy_and_hold_bij_stijgende_prijs_geeft_winst` | Gegeven een stijgende prijs, wanneer je koopt en vasthoudt, dan maak je winst. | Acceptatie | ✅ geslaagd | 10 ms |
| `test_eindrapport_metrics_zijn_geldig` | Na een episode levert de env bruikbare, geldige metrics op voor rapportage. | Acceptatie | ✅ geslaagd | 10 ms |
| `test_flat_env_werkt_end_to_end_voor_mlp` | De flat-variant (voor MLP) levert een platte vector en draait een episode af. | Acceptatie | ✅ geslaagd | 3 ms |
| `test_volledige_episode_loopt_zonder_fouten` | Gegeven een env, wanneer een agent een hele episode speelt, dan eindigt die netjes. | Acceptatie | ✅ geslaagd | 10 ms |

### `test_deeplob`

_DeepLOB-model — outputvorm, NaN-check, determinisme, gradiënten_

**Soort:** Unit test (component/smoke)

Test het neurale netwerk geïsoleerd: één forward/backward pass op random input, controleert vorm en numerieke gezondheid.

| Test | Wat wordt getest | Soort | Uitslag | Tijd |
|---|---|---|---|---|
| `test_deterministisch_in_eval` | In eval-mode geeft dezelfde input twee keer dezelfde output. | Robuustheid | ✅ geslaagd | 36 ms |
| `test_geen_nan_in_output` | De output bevat geen NaN-waarden (numeriek stabiel). | Robuustheid | ✅ geslaagd | 12 ms |
| `test_gradient_stroomt` | Backprop werkt: na loss.backward() krijgt minstens één parameter een gradient. | Robuustheid | ✅ geslaagd | 2441 ms |
| `test_output_vorm` | De forward pass geeft de juiste outputvorm (batch, output_dim) = (4, 64). | Contract | ✅ geslaagd | 10 ms |

### `test_features`

_Feature engineering — spread, order imbalance, RSI, returns, momentum_

**Soort:** Unit test

Test de pure functie `add_features` op een kleine, handgemaakte DataFrame met verifieerbare uitkomsten.

| Test | Wat wordt getest | Soort | Uitslag | Tijd |
|---|---|---|---|---|
| `test_momentum_positief_bij_stijgende_prijs` | Bij stijgende prijs is momentum_10 positief. | Gedrag | ✅ geslaagd | 11 ms |
| `test_order_imbalance_bereik` | order_imbalance = (bid_vol − ask_vol)/(bid_vol + ask_vol), hier 1/3. | Numeriek | ✅ geslaagd | 12 ms |
| `test_returns_positief_bij_stijgende_prijs` | Bij stijgende prijs is de 5s-return positief. | Gedrag | ✅ geslaagd | 8 ms |
| `test_rsi_hoog_bij_stijgende_prijs` | Bij monotoon stijgende prijs ligt RSI dicht bij 100 (>90). | Gedrag | ✅ geslaagd | 6 ms |
| `test_rsi_in_bereik_0_100` | RSI ligt altijd binnen het geldige bereik [0, 100]. | Numeriek | ✅ geslaagd | 8 ms |
| `test_spread_is_ask_min_bid` | spread = ask − bid, hier constant 1.0. | Numeriek | ✅ geslaagd | 8 ms |

### `test_performance`

_Non-functioneel — throughput, latency, geheugen-efficiëntie_

**Soort:** Non-functionele test

Test kwaliteitseisen i.p.v. functionaliteit: snelheid (steps/s, inference-latency) en zuinig geheugen (zero-copy sliding window).

| Test | Wat wordt getest | Soort | Uitslag | Tijd |
|---|---|---|---|---|
| `test_env_bewaart_ruwe_array_zonder_kopie` | De env bewaart de ruwe (N, features) array, niet een uitgerolde kopie. | Geheugen | ✅ geslaagd | 0 ms |
| `test_sliding_window_is_zero_copy` | Een window-slice deelt geheugen met de ruwe array (geen kopie). | Geheugen | ✅ geslaagd | 1 ms |
| `test_streaming_veel_zuiniger_dan_materialiseren` | Streaming-opslag is ~window_size keer kleiner dan uitgerolde sequences. | Geheugen | ✅ geslaagd | 0 ms |
| `test_deeplob_inference_latency` | DeepLOB verwerkt een batch van 32 in minder dan 2 seconden (CPU). | Performance | ✅ geslaagd | 16 ms |
| `test_env_throughput` | De env verwerkt minstens 2000 steps/seconde (ruime ondergrens). | Performance | ✅ geslaagd | 452 ms |

### `test_trading_env`

_CryptoTradingEnv — reset, step, kopen/verkopen, portfolio, observaties_

**Soort:** Unit test

Test de environment-klasse geïsoleerd met synthetische data (constante prijs, geen echte dataset of model nodig).

| Test | Wat wordt getest | Soort | Uitslag | Tijd |
|---|---|---|---|---|
| `test_info_bevat_kernmetrics` | De info-dict bevat de kernmetrics (portfolio_value, return, trades, drawdown). | Contract | ✅ geslaagd | 0 ms |
| `test_kosten_verlagen_portfolio_bij_constante_prijs` | Bij constante prijs verlaagt de transactiekost (flat_fee) de portfoliowaarde. | Numeriek | ✅ geslaagd | 0 ms |
| `test_portfoliowaarde_bij_start` | De portfoliowaarde bij start is gelijk aan het startkapitaal. | Numeriek | ✅ geslaagd | 0 ms |
| `test_observatie_dtype_float32` | Beide observatie-arrays zijn van type float32 (klaar voor PyTorch). | Contract | ✅ geslaagd | 0 ms |
| `test_observatie_vorm` | De observatie heeft de juiste vorm: features (50,15) en portfolio (4,). | Contract | ✅ geslaagd | 0 ms |
| `test_reset_geeft_startkapitaal` | Na reset staat de balans op het startkapitaal en is er geen BTC. | Gedrag | ✅ geslaagd | 0 ms |
| `test_buy_dan_sell_sluit_positie` | Buy gevolgd door Sell (2) sluit de positie: btc_held terug naar 0. | Gedrag | ✅ geslaagd | 0 ms |
| `test_buy_zet_cash_om_in_btc` | Actie Buy (1) zet cash om in BTC: btc_held stijgt, balans daalt. | Gedrag | ✅ geslaagd | 0 ms |
| `test_episode_eindigt_aan_einde_data` | De episode eindigt (truncated=True) zodra de data op is. | Gedrag | ✅ geslaagd | 2 ms |
| `test_hold_verandert_balans_niet` | Actie Hold (0) doet geen transactie: balans en BTC blijven gelijk. | Gedrag | ✅ geslaagd | 0 ms |
| `test_step_geeft_vijf_waarden` | step() geeft de Gymnasium-tuple terug: (obs, reward, terminated, truncated, info). | Contract | ✅ geslaagd | 1 ms |

## Code-coverage

Gemeten met **coverage.py** over de domeinlogica (`src`, `dataVerwerken`). Coverage = het percentage broncode-regels dat tijdens de tests is uitgevoerd.

> Bewust buiten scope: CLI-trainingsscripts (`train/`), logging/plotting-utilities (`*/utils/*`), `vec_env.py` (subprocessen) en `create_core_data.py`. Dat zijn entry-points/infra die via acceptatie- en integratietests worden gedekt, niet via unit tests.

**Totaal (domeinlogica): 29.2%**

| Bestand | Regels | Gedekt | Coverage |
|---|---|---|---|
| `dataVerwerken/preprocess_data.py` | 358 | 73 | 20% |
| `src/models/ppo.py` | 352 | 34 | 10% |
| `src/models/sac.py` | 324 | 37 | 11% |
| `src/envs/trading_env.py` | 296 | 209 | 71% |
| `src/models/mlp.py` | 166 | 29 | 17% |
| `src/models/deeplob.py` | 142 | 90 | 63% |
| `src/models/__init__.py` | 5 | 5 | 100% |
| `src/envs/__init__.py` | 2 | 2 | 100% |
| `src/__init__.py` | 1 | 1 | 100% |
