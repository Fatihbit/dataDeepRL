"""
Test-runner met documentatie-generatie.

Draait alle tests in deze map en schrijft een leesbaar rapport naar
tests/TEST_REPORT.md. Het rapport documenteert per test wát er getest is
(uit de docstring) én de uitslag (geslaagd / gefaald / error).

Gebruik:
    python -m tests.run_tests
    python tests/run_tests.py
"""

import io
import os
import sys
import time
import unittest
from datetime import datetime

try:
    import coverage  # code-coverage tool (pip install coverage)
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_DIR)

REPORT_PATH = os.path.join(TESTS_DIR, "TEST_REPORT.md")

# Broncode die we meten voor coverage: de domeinlogica (env, modellen, features).
# We sluiten bewust uit:
#  - train/        : CLI trainingsscripts (entry-points; gevalideerd via acceptatie/integratie)
#  - */utils/*     : logging/plotting/callbacks (I/O-infra, geen unit-test-doel)
#  - vec_env.py    : draait subprocessen, niet zinvol als unit te testen
#  - create_core_data.py : eenmalig data-pipeline CLI-script
COVERAGE_SOURCE = ["src", "dataVerwerken"]
COVERAGE_OMIT = [
    "*/tests/*",
    "*/utils/*",
    "*/vec_env.py",
    "*/create_core_data.py",
]

# Metadata per testmodule: onderwerp + soort test + uitleg waarom het dat soort is.
MODULE_META = {
    "test_trading_env": {
        "onderwerp": "CryptoTradingEnv — reset, step, kopen/verkopen, portfolio, observaties",
        "soort": "Unit test",
        "uitleg": "Test de environment-klasse geïsoleerd met synthetische data "
                  "(constante prijs, geen echte dataset of model nodig).",
    },
    "test_features": {
        "onderwerp": "Feature engineering — spread, order imbalance, RSI, returns, momentum",
        "soort": "Unit test",
        "uitleg": "Test de pure functie `add_features` op een kleine, "
                  "handgemaakte DataFrame met verifieerbare uitkomsten.",
    },
    "test_deeplob": {
        "onderwerp": "DeepLOB-model — outputvorm, NaN-check, determinisme, gradiënten",
        "soort": "Unit test (component/smoke)",
        "uitleg": "Test het neurale netwerk geïsoleerd: één forward/backward pass "
                  "op random input, controleert vorm en numerieke gezondheid.",
    },
    "test_performance": {
        "onderwerp": "Non-functioneel — throughput, latency, geheugen-efficiëntie",
        "soort": "Non-functionele test",
        "uitleg": "Test kwaliteitseisen i.p.v. functionaliteit: snelheid (steps/s, "
                  "inference-latency) en zuinig geheugen (zero-copy sliding window).",
    },
    "test_acceptance": {
        "onderwerp": "Acceptatie — end-to-end gebruikersscenario's (Given-When-Then)",
        "soort": "Acceptatietest (E2E)",
        "uitleg": "Test of complete scenario's vanuit gebruikersperspectief werken: "
                  "een hele episode spelen, geldige rapport-metrics, buy-and-hold winst.",
    },
}

# Subcategorie per test (op naam). Bepaalt wát voor soort check het is.
TEST_CATEGORIE = {
    # trading env
    "test_reset_geeft_startkapitaal": "Gedrag",
    "test_observatie_vorm": "Contract",
    "test_observatie_dtype_float32": "Contract",
    "test_hold_verandert_balans_niet": "Gedrag",
    "test_step_geeft_vijf_waarden": "Contract",
    "test_buy_zet_cash_om_in_btc": "Gedrag",
    "test_buy_dan_sell_sluit_positie": "Gedrag",
    "test_episode_eindigt_aan_einde_data": "Gedrag",
    "test_portfoliowaarde_bij_start": "Numeriek",
    "test_kosten_verlagen_portfolio_bij_constante_prijs": "Numeriek",
    "test_info_bevat_kernmetrics": "Contract",
    # features
    "test_spread_is_ask_min_bid": "Numeriek",
    "test_order_imbalance_bereik": "Numeriek",
    "test_rsi_in_bereik_0_100": "Numeriek",
    "test_rsi_hoog_bij_stijgende_prijs": "Gedrag",
    "test_returns_positief_bij_stijgende_prijs": "Gedrag",
    "test_momentum_positief_bij_stijgende_prijs": "Gedrag",
    # deeplob
    "test_output_vorm": "Contract",
    "test_geen_nan_in_output": "Robuustheid",
    "test_deterministisch_in_eval": "Robuustheid",
    "test_gradient_stroomt": "Robuustheid",
    # performance (non-functioneel)
    "test_env_throughput": "Performance",
    "test_deeplob_inference_latency": "Performance",
    "test_sliding_window_is_zero_copy": "Geheugen",
    "test_env_bewaart_ruwe_array_zonder_kopie": "Geheugen",
    "test_streaming_veel_zuiniger_dan_materialiseren": "Geheugen",
    # acceptatie (end-to-end)
    "test_volledige_episode_loopt_zonder_fouten": "Acceptatie",
    "test_eindrapport_metrics_zijn_geldig": "Acceptatie",
    "test_buy_and_hold_bij_stijgende_prijs_geeft_winst": "Acceptatie",
    "test_flat_env_werkt_end_to_end_voor_mlp": "Acceptatie",
}

# Uitleg van de gebruikte categorieën (legenda in het rapport).
CATEGORIE_LEGENDA = [
    ("Unit test", "Test één component/functie geïsoleerd, zonder externe data of andere modules."),
    ("Gedrag", "Controleert of een actie het juiste gevolg heeft (bv. kopen verlaagt de balans)."),
    ("Contract", "Controleert vorm, datatype of return-signatuur (bv. observatie is (50,15) float32)."),
    ("Numeriek", "Controleert een exacte waarde of geldig bereik (bv. spread = 1.0, RSI in [0,100])."),
    ("Robuustheid", "Controleert numerieke gezondheid (geen NaN, determinisme, gradiënten stromen)."),
    ("Performance", "Non-functioneel: snelheid moet boven een drempel liggen (steps/s, latency)."),
    ("Geheugen", "Non-functioneel: data wordt zuinig opgeslagen (zero-copy sliding window)."),
    ("Acceptatie", "End-to-end: een compleet gebruikersscenario werkt van begin tot eind."),
]


class DocResult(unittest.TestResult):
    """TestResult die per test de uitslag, tijd en docstring vasthoudt."""

    def __init__(self):
        super().__init__()
        self.records = []  # lijst van dicts: module, naam, doc, status
        self._start = None

    def startTest(self, test):
        self._start = time.time()
        super().startTest(test)

    def _record(self, test, status, detail=""):
        duur = time.time() - self._start if self._start else 0.0
        self.records.append({
            "module": test.__class__.__module__,
            "klasse": test.__class__.__name__,
            "naam": test._testMethodName,
            "doc": (test.shortDescription() or "(geen beschrijving)").strip(),
            "status": status,
            "duur": duur,
            "detail": detail,
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "✅ geslaagd")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "❌ gefaald", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "⚠️ error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "⏭️ overgeslagen", reason)


def _meet_coverage(cov):
    """Verzamel coverage: (totaal_pct, [(bestand, regels, gedekt, pct), ...])."""
    cov.stop()
    totaal_pct = cov.report(file=io.StringIO())  # totaal-percentage
    data = cov.get_data()
    per_bestand = []
    for f in sorted(data.measured_files()):
        try:
            _, statements, _, missing, _ = cov.analysis2(f)
        except Exception:
            continue
        n = len(statements)
        if n == 0:
            continue
        gedekt = n - len(missing)
        rel = os.path.relpath(f, PROJECT_DIR).replace("\\", "/")
        per_bestand.append((rel, n, gedekt, gedekt / n * 100))
    return totaal_pct, per_bestand


def run() -> bool:
    """Draai alle tests, schrijf het rapport, en geef True als alles slaagt."""
    cov = None
    if HAS_COVERAGE:
        cov = coverage.Coverage(source=COVERAGE_SOURCE, omit=COVERAGE_OMIT)
        cov.start()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TESTS_DIR, pattern="test_*.py")

    result = DocResult()
    start = time.time()
    suite.run(result)
    elapsed = time.time() - start

    coverage_info = None
    if cov is not None:
        try:
            coverage_info = _meet_coverage(cov)
        except Exception as exc:  # coverage mag het rapport nooit laten falen
            print(f"  (coverage overgeslagen: {exc})")

    total = result.testsRun
    failed = len(result.failures)
    errored = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - errored - skipped

    _write_report(result, total, passed, failed, errored, skipped, elapsed,
                  coverage_info)

    # Console samenvatting
    print(f"\n{'='*55}")
    print(f"  Tests: {total} | Geslaagd: {passed} | Gefaald: {failed} "
          f"| Errors: {errored} | Overgeslagen: {skipped}")
    if coverage_info:
        print(f"  Coverage: {coverage_info[0]:.1f}%")
    print(f"  Tijd: {elapsed:.2f}s")
    print(f"  Rapport: {REPORT_PATH}")
    print(f"{'='*55}")

    return failed == 0 and errored == 0


def _write_report(result, total, passed, failed, errored, skipped, elapsed,
                  coverage_info=None):
    status = "✅ ALLE TESTS GESLAAGD" if (failed == 0 and errored == 0) else "❌ ER ZIJN FOUTEN"
    pct = (passed / total * 100) if total else 0.0

    lines = [
        "# Testrapport & Testdocumentatie — DataDeepRL",
        "",
        f"**Gegenereerd:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## {status}",
        "",
        "## Samenvatting",
        "",
        "| Metric | Waarde |",
        "|---|---|",
        f"| Totaal tests | {total} |",
        f"| Geslaagd | {passed} |",
        f"| Gefaald | {failed} |",
        f"| Errors | {errored} |",
        f"| Overgeslagen | {skipped} |",
        f"| Slaagpercentage | {pct:.1f}% |",
    ]
    if coverage_info:
        lines.append(f"| Code-coverage | {coverage_info[0]:.1f}% |")
    lines += [
        f"| Duur | {elapsed:.2f}s |",
        "",
    ]

    # Groepeer records per module.
    per_module = {}
    for rec in result.records:
        per_module.setdefault(rec["module"], []).append(rec)

    # Soorten tests (legenda) — legt uit wat de gebruikte categorieën betekenen.
    lines += ["## Soorten tests", "",
              "De testsuite volgt de **testpiramide**: veel snelle unit tests onderaan, "
              "aangevuld met non-functionele tests (performance/geheugen) en een laag "
              "acceptatietests (end-to-end) bovenaan. Alle tests gebruiken synthetische "
              "data en hebben geen echte dataset of getraind model nodig. Per test geeft "
              "de kolom **Soort** aan wat voor soort controle het is:", "",
              "| Soort | Betekenis |", "|---|---|"]
    for naam, uitleg in CATEGORIE_LEGENDA:
        lines.append(f"| **{naam}** | {uitleg} |")
    lines.append("")

    # Overzicht per module.
    lines += ["## Geteste onderdelen", "",
              "| Module | Soort | Onderwerp | Tests |", "|---|---|---|---|"]
    for module in sorted(per_module):
        meta = MODULE_META.get(module, {})
        lines.append(
            f"| `{module}` | {meta.get('soort', 'Unit test')} "
            f"| {meta.get('onderwerp', '')} | {len(per_module[module])} |"
        )
    lines.append("")

    # Gedetailleerde documentatie: per test wat getest is + soort + uitslag.
    lines += ["## Wat is er getest", ""]
    for module in sorted(per_module):
        meta = MODULE_META.get(module, {})
        lines += [f"### `{module}`", ""]
        if meta.get("onderwerp"):
            lines += [f"_{meta['onderwerp']}_", ""]
        lines += [f"**Soort:** {meta.get('soort', 'Unit test')}", ""]
        if meta.get("uitleg"):
            lines += [f"{meta['uitleg']}", ""]
        lines += ["| Test | Wat wordt getest | Soort | Uitslag | Tijd |",
                  "|---|---|---|---|---|"]
        for rec in per_module[module]:
            categorie = TEST_CATEGORIE.get(rec["naam"], "—")
            lines.append(
                f"| `{rec['naam']}` | {rec['doc']} | {categorie} "
                f"| {rec['status']} | {rec['duur']*1000:.0f} ms |"
            )
        lines.append("")

    # Code-coverage sectie.
    if coverage_info:
        totaal_pct, per_bestand = coverage_info
        lines += ["## Code-coverage", "",
                  f"Gemeten met **coverage.py** over de domeinlogica "
                  f"(`{'`, `'.join(COVERAGE_SOURCE)}`). Coverage = het percentage "
                  f"broncode-regels dat tijdens de tests is uitgevoerd.",
                  "",
                  "> Bewust buiten scope: CLI-trainingsscripts (`train/`), logging/plotting-"
                  "utilities (`*/utils/*`), `vec_env.py` (subprocessen) en `create_core_data.py`. "
                  "Dat zijn entry-points/infra die via acceptatie- en integratietests worden "
                  "gedekt, niet via unit tests.",
                  "",
                  f"**Totaal (domeinlogica): {totaal_pct:.1f}%**", "",
                  "| Bestand | Regels | Gedekt | Coverage |", "|---|---|---|---|"]
        # Toon de bestanden met de meeste regels eerst (meest relevant).
        for rel, n, gedekt, p in sorted(per_bestand, key=lambda x: -x[1])[:20]:
            lines.append(f"| `{rel}` | {n} | {gedekt} | {p:.0f}% |")
        lines.append("")

    # Detail bij fouten/errors.
    probleem = [r for r in result.records if r["status"] not in ("✅ geslaagd", "⏭️ overgeslagen")]
    if probleem:
        lines += ["## Details van fouten", ""]
        for rec in probleem:
            lines += [
                f"### {rec['status']} — `{rec['module']}.{rec['klasse']}.{rec['naam']}`",
                "",
                "```",
                rec["detail"].strip(),
                "```",
                "",
            ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
