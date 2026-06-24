# Teststrategie — DataDeepRL

Dit document beschrijft hoe kwaliteit in DataDeepRL geborgd wordt: welke
soorten tests er zijn, met welke testtechnieken ze zijn opgesteld, hoe ze
geautomatiseerd draaien (ontwikkelstraat), hoe code-coverage wordt gemeten en
hoe code-reviews zijn ingezet.

---

## 1. Testpiramide & soorten tests

We volgen de klassieke **testpiramide**: veel snelle, goedkope tests onderaan
en weinig dure tests bovenaan.

```
              ┌───────────────────────────┐
              │   Acceptatietests (E2E)    │   4 tests  — gebruikersscenario's
              ├───────────────────────────┤
              │ Non-functionele tests      │   5 tests  — performance & geheugen
              ├───────────────────────────┤
              │        Unit tests          │  21 tests  — losse componenten
              └───────────────────────────┘
```

| Laag | Bestand | Wat | Aantal |
|---|---|---|---|
| Unit | `tests/test_trading_env.py` | Environment-logica (reset/step/portfolio) | 11 |
| Unit | `tests/test_features.py` | Feature engineering (formules) | 6 |
| Unit | `tests/test_deeplob.py` | DeepLOB-model (forward/backward) | 4 |
| Non-functioneel | `tests/test_performance.py` | Throughput, latency, geheugen | 5 |
| Acceptatie | `tests/test_acceptance.py` | End-to-end scenario's (Given-When-Then) | 4 |

Alle tests draaien op **synthetische data** — geen `coreData/` of getraind
model nodig — zodat ze overal en in CI snel en deterministisch zijn.

---

## 2. Gebruikte testtechnieken

Tests zijn niet ad hoc bedacht maar afgeleid met erkende technieken:

### Equivalentieklassen (equivalence partitioning)
De acties van de agent vallen in drie klassen — **Hold (0)**, **Buy (1)**,
**Sell (2)** — die elk apart getest worden
(`test_hold_verandert_balans_niet`, `test_buy_zet_cash_om_in_btc`,
`test_buy_dan_sell_sluit_positie`). Eén representatieve test per klasse i.p.v.
alle mogelijke acties.

### Grenswaarde-analyse (boundary value analysis)
- **RSI** wordt getest op de grenzen van zijn geldige bereik `[0, 100]`
  (`test_rsi_in_bereik_0_100`) en op het extreme geval van een monotoon
  stijgende prijs → RSI ≈ 100 (`test_rsi_hoog_bij_stijgende_prijs`).
- Het **einde van de data** is een grensgeval: de episode moet exact dan
  stoppen (`test_episode_eindigt_aan_einde_data`).

### Specificatiegebaseerd testen (formule-verificatie)
Features met een vaste wiskundige definitie worden tegen een handberekende
waarde gelegd: `spread = ask − bid = 1.0`, `order_imbalance = 1/3`. De
verwachte waarde volgt uit de formule, niet uit de implementatie.

### Eigenschap-gebaseerd redeneren (property/metamorphic)
In plaats van exacte uitkomsten testen we invarianten die altijd moeten
gelden, ongeacht de (random) input:
- portfoliowaarde is altijd eindig en ≥ 0;
- `win_rate ∈ [0, 1]`;
- reward is altijd eindig.
- **Metamorf**: een *stijgende* prijs hoort bij buy-and-hold een *positieve*
  return te geven (`test_buy_and_hold_bij_stijgende_prijs_geeft_winst`).

### Contracttesten (interface/shape)
De observatie moet voldoen aan het Gymnasium-contract: juiste vorm
`(window, features)`, dtype `float32`, en `step()` geeft een 5-tuple terug.

### Non-functioneel testen met drempelwaarden
Performance/geheugen worden getest met **ruime ondergrenzen** (bv. > 2000
steps/s) zodat ze regressies vangen zonder flaky te worden op trage machines.

---

## 3. Non-functionele eisen (kwaliteitsattributen)

| Eis | Test | Drempel |
|---|---|---|
| Doorvoer env | `test_env_throughput` | > 2000 steps/s |
| Inference-latency | `test_deeplob_inference_latency` | batch 32 < 2 s (CPU) |
| Zero-copy windowing | `test_sliding_window_is_zero_copy` | window deelt geheugen met bron |
| Geen data-duplicatie | `test_env_bewaart_ruwe_array_zonder_kopie` | env bewaart de ruwe array |
| Geheugen-efficiëntie | `test_streaming_veel_zuiniger_dan_materialiseren` | ~window_size × zuiniger |

Deze tests borgen de **efficiënte algoritmiek & datastructuur** van het
project: de environment rolt de sliding windows *on-the-fly* uit via
zero-copy numpy-slicing, in plaats van een `(N × seq_len × num_features)`
array te materialiseren (≈ 100× meer geheugen).

---

## 4. Ontwikkelstraat (CI/CD) & automatisering

De volledige suite draait **geautomatiseerd** via één commando:

```bash
python -m tests.run_tests
```

Dit:
1. detecteert automatisch alle `test_*.py` (unittest discovery);
2. meet code-coverage met **coverage.py**;
3. genereert [`tests/TEST_REPORT.md`](tests/TEST_REPORT.md) met per test:
   wát er getest is, het soort test en de uitslag;
4. geeft exitcode 0/1 — bruikbaar als CI-poort.

**GitHub Actions** ([.github/workflows/tests.yml](.github/workflows/tests.yml))
draait deze straat bij elke push en pull request naar `main`:
checkout → Python opzetten → dependencies → tests + coverage → testrapport als
build-artefact uploaden. Een gefaalde test laat de build falen, zodat kapotte
code niet op `main` belandt.

### Gebruikte tools
| Doel | Tool |
|---|---|
| Testframework | `unittest` (standaardbibliotheek, geen extra dependency) |
| Code-coverage | `coverage.py` |
| Numerieke asserts | `numpy.testing`, `torch.testing` |
| CI/CD | GitHub Actions |
| Rapportage | eigen generator in `tests/run_tests.py` (Markdown) |

---

## 5. Code-coverage

Coverage wordt bij elke run gemeten en in het rapport getoond, met een
per-bestand uitsplitsing. We meten bewust de **domeinlogica** (`src`,
`dataVerwerken`) en sluiten entry-points/infra uit:

| Buiten scope | Reden |
|---|---|
| `train/` | CLI-trainingsscripts; gevalideerd via acceptatie/integratie, niet via unit tests |
| `*/utils/*` | logging, plotting, callbacks — I/O-infrastructuur |
| `vec_env.py` | start subprocessen, niet zinvol als unit te testen |
| `create_core_data.py` | eenmalig data-pipeline CLI-script |

De kernmodules zijn goed gedekt (env ≈ 71%, DeepLOB ≈ 63%). De lagere
totaalscore komt door de grote, nog niet unit-geteste algoritme-modules
(`ppo.py`, `sac.py`) — dit is een bewuste, gedocumenteerde keuze en een
duidelijk volgend uitbreidingspunt.

---

## 6. Code-reviews

- **Geautomatiseerde review**: `/code-review ultra` (multi-agent cloud review
  van de branch/PR) wordt ingezet vóór merge naar `main`.
- **PR-flow**: wijzigingen lopen via een pull request naar `main`; de CI moet
  groen zijn (alle tests geslaagd) voordat er gemergd wordt.
- **Reviewfocus**: correctheid van de RL-/trading-logica, numerieke
  stabiliteit (NaN/inf, clipping) en geheugengedrag (geen onnodige kopieën).

---

## 7. Geavanceerde technieken

- **Zero-copy verificatie** met `numpy.shares_memory` om te bewijzen dat de
  sliding window geen geheugen dupliceert.
- **Metamorphic testing**: relatie tussen input (stijgende prijs) en output
  (positieve return) i.p.v. een exacte waarde.
- **Property-based invarianten** op random episodes (eindigheid, geldige
  bereiken) i.p.v. één vast scenario.
- **Determinisme-test** van het neurale net in `eval`-mode (zelfde input →
  zelfde output), wat reproduceerbaarheid garandeert.

---

## 8. Reproduceren

```bash
pip install -r requirements.txt          # incl. coverage
python -m tests.run_tests                # draait alles + genereert rapport
```

Het resultaat staat in [`tests/TEST_REPORT.md`](tests/TEST_REPORT.md).
