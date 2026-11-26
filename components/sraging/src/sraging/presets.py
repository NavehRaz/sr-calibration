"""Legacy presets' theta, read without the legacy traps (spec 005 US7; FR-024; research R9; data-model "Preset";
register D-502).

    params, info = get("mice_F")                       # SRParams in days; info.kappa_rule, info.time_unit == "days"
    convert(params, to="weeks")                        # units only when asked (sraging.units)

The values are SRK's `Preset_values/summery_{mode_overall,mode,max_likelihood}.csv`, packaged as byte copies of the
frozen snapshot's (src/sraging/data/presets_legacy/, MANIFEST.json; the library never reads tests/oracle), read with
getTheta's arithmetic (tests/oracle/fp_piecewise_cb191f5/SRtools_kappa/SRK/presets.py:256-480):
- eta = xc / (xc/eta), beta = (beta/eta) eta, epsilon = xc^2 / (xc^2/epsilon), from the ratio rows;
- `_kappa_convention` (:227-253): no file has a `kappa` row, so every preset is the legacy restatement X -> X / xc,
  kappa = 0.5 / xc, xc = 1 (a restatement of Karin's model, not a fit);
- days, the tables' unit, always: getTheta's time_unit="auto" picked the unit from a substring of the name (mice in
  weeks, D-502). `sraging.units.convert` converts when asked;
- ExtH=True adds the external hazard. The table's ExtH was fitted in the preset's "auto" unit (`auto_unit`, getTheta's
  rule, :391-403; years for all 16 presets that have one), and getTheta returned it raw at every unit (D-502). SRK's
  convention is the rate exp(-ExtH) per unit time (a finite external_hazard kills with probability
  exp(-external_hazard) dt, SR_hetro.py:460-462; spec 004's presets, in the auto unit), so in days
  h_ext = exp(-ExtH) / UNITS[auto unit], and `convert(params, to=info.ext_h_unit).h_ext` is exp(-ExtH). A preset
  without one (NaN) has h_ext = 0. The raw ExtH is `PresetInfo.ext_h`, its unit `PresetInfo.ext_h_unit`.
- names: the tables' preset columns; aliases: an explicit copy of SRK's PRESET_ALIASES (:16-127), organism names only.
  A parameter-row name ('beta/xc', 'k/xc', ...), an alias whose target is not a column, and any source but "legacy"
  are refused by name.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams
from sraging.units import UNITS

SOURCES = ("legacy",)
FILES = MappingProxyType({"mode_overall": "summery_mode_overall.csv", "mode": "summery_mode.csv",
                          "max_likelihood": "summery_max_likelihood.csv"})
KAPPA_RULE = "0.5/xc (legacy restatement)"
#: SRK's PRESET_ALIASES (presets.py:16-127), copied verbatim
ALIASES = MappingProxyType({
    'Mice_M': 'mice_M', 'mice_m': 'mice_M', 'Mice M': 'mice_M', 'mice M': 'mice_M', 'mice m': 'mice_M',
    'male mice': 'mice_M', 'Male Mice': 'mice_M', 'Mice_F': 'mice_F', 'mice_f': 'mice_F', 'Mice F': 'mice_F',
    'mice F': 'mice_F', 'mice f': 'mice_F', 'female mice': 'mice_F', 'Female Mice': 'mice_F', 'cats': 'cats_BPH',
    'Cats': 'cats_BPH', 'drosophila': 'drosophila_441', 'Drosophila': 'drosophila_441', 'flies': 'drosophila_441',
    'drosophila melanogaster': 'drosophila_441', 'Drosophila Melanogaster': 'drosophila_441',
    'humans_M': 'combined_human_M', 'Humans M': 'combined_human_M', 'Humans_M': 'combined_human_M',
    'humans male': 'combined_human_M', 'Humans Male': 'combined_human_M', 'humans_M_combined': 'combined_human_M',
    'humans_F': 'combined_human_F', 'Humans F': 'combined_human_F', 'Humans_F': 'combined_human_F',
    'humans female': 'combined_human_F', 'Humans Female': 'combined_human_F',
    'humans_F_combined': 'combined_human_F', 'Denmark_M_1900': 'Denmark_M_1900_hetro',
    'Denmark_M_1890': 'Denmark_M_1890_hetro', 'Denmark_F_1900': 'Denmark_F_1900_hetro',
    'Denmark_F_1890': 'Denmark_F_1890_hetro', 'Labradors': 'Labradors_vetCompass', 'dogs': 'Labradors_vetCompass',
    'Dogs': 'Labradors_vetCompass', 'dog labrador': 'Labradors_vetCompass', 'Dog Labrador': 'Labradors_vetCompass',
    'Staffy': 'Staffy_vetCompass', 'staffy': 'Staffy_vetCompass', 'Staffy_vetCompass': 'Staffy_vetCompass',
    'staffy_vetCompass': 'Staffy_vetCompass', 'dog staffy': 'Staffy_vetCompass', 'Dog Staffy': 'Staffy_vetCompass',
    'German Shepherd': 'GermanShepherd_vetCompass', 'German Shepherd_vetCompass': 'GermanShepherd_vetCompass',
    'german shepherd': 'GermanShepherd_vetCompass', 'dog german shepherd': 'GermanShepherd_vetCompass',
    'Dog German Shepherd': 'GermanShepherd_vetCompass', 'Jack Russell': 'Jack_Russell_vetCompass',
    'Jack Russell_vetCompass': 'Jack_Russell_vetCompass', 'jack russell': 'Jack_Russell_vetCompass',
    'dog jack russell': 'Jack_Russell_vetCompass', 'Dog Jack Russell': 'Jack_Russell_vetCompass',
    'C. elegans': 'celegans', 'c. elegans': 'celegans', 'Celegans': 'celegans', 'celegans': 'celegans',
    'E. coli': 'ecoli', 'e. coli': 'ecoli', 'Ecoli': 'ecoli', 'E coli': 'ecoli', 'Yeast': 'yeast',
    'Budding yeast': 'yeast', 'budding yeast': 'yeast', 'Guinea pigs': 'Guiniea_pig_VC',
    'guinea pigs': 'Guiniea_pig_VC', 'Guinea Pig': 'Guiniea_pig_VC', 'guinea pig': 'Guiniea_pig_VC',
    'gpig': 'Guiniea_pig_VC', 'Guinea Pig VC': 'Guiniea_pig_VC', 'guinea pig vc': 'Guiniea_pig_VC',
})
#: getTheta's time_unit="auto" rule (presets.py:391-403), in its order: the first unit with a keyword that is a
#: substring of the lower-cased table name (after the alias, :386-389); else days
AUTO_UNIT_KEYWORDS = (
    ("years", ('human', 'sweden', 'denmark', 'dog', 'cat', 'labrador', 'staffy', 'german', 'jack', 'pig',
               'guinea pigs')),
    ("hours", ('ecoli', 'e. coli')),
    ("weeks", ('mice', 'mouse')),
    ("generations", ('yeast',)),
)


def _dir():
    return resources.files("sraging") / "data" / "presets_legacy"


_TABLES: dict = {}


def _table(type_):
    """{row: {preset: float}} of one summary file, parsed with float() (equal to SRK's pandas read on every cell of the
    three packaged files: tests/unit/test_presets.py)."""
    if type_ not in _TABLES:
        rows = list(csv.reader((_dir() / FILES[type_]).read_text().splitlines()))
        head = rows[0]
        out = {}
        for r in rows[1:]:
            vals = {}
            for col, cell in zip(head[1:], r[1:]):
                if col.endswith("95% CI"):
                    continue
                try:
                    vals[col] = float(cell) if cell.strip() else math.nan
                except ValueError:
                    vals[col] = math.nan
            out[r[0]] = vals
        _TABLES[type_] = out
    return _TABLES[type_]


def names(*, type="mode_overall") -> tuple:
    """The presets in a summary file, in its column order."""
    _type(type)
    return tuple(_table(type)["xc"])


def _type(type_):
    if type_ not in FILES:
        raise UnsupportedOptionError("type", f"type is one of {list(FILES)}, got {type_!r}")


def auto_unit(name) -> str:
    """The unit getTheta's time_unit="auto" gives the table column `name` (AUTO_UNIT_KEYWORDS): the unit its ExtH
    was fitted in."""
    low = name.lower()
    return next((u for u, words in AUTO_UNIT_KEYWORDS if any(w in low for w in words)), "days")


@dataclass(frozen=True)
class PresetInfo:
    """What a preset is: its table name, the name asked for, the summary type, the source, whether ExtH was asked, its
    raw table value and the unit it is in (getTheta's auto unit; both None when not asked or the table has none), the
    kappa rule, the unit, and the file's sha256."""

    name: str
    requested: str
    type: str
    source: str
    ExtH: bool
    ext_h: float | None
    ext_h_unit: str | None
    kappa_rule: str
    time_unit: str
    file_sha256: str


def get(name, *, source="legacy", type="mode_overall", ExtH=False) -> tuple:
    """(SRParams in days, PresetInfo) for a legacy preset (FR-024)."""
    if source not in SOURCES:
        raise UnsupportedOptionError("source", f"source {source!r}: only 'legacy' (SRK's Preset_values) is read in "
                                               f"spec 005; presets calibrated with sraging are [T-LANGEVIN-PRESETS]")
    _type(type)
    if not isinstance(ExtH, bool):
        raise UnsupportedOptionError("ExtH", f"ExtH must be True or False, got {ExtH!r}")
    tab = _table(type)
    if not isinstance(name, str):
        raise UnsupportedOptionError("name", f"a preset name is a string, got {name!r}")
    if name in tab:
        raise UnsupportedOptionError("name", f"{name!r} is a parameter row of the preset tables, not a preset")
    col = ALIASES.get(name, name)
    if col not in tab["xc"]:
        why = (f"alias {name!r} names {col!r}, which is not a preset in {FILES[type]}" if col != name else
               f"no preset {name!r} in {FILES[type]}")
        raise UnsupportedOptionError("name", f"{why}; presets: {', '.join(names(type=type))}")
    xc_eta, beta_eta, xc2_eps, xc = (tab[r][col] for r in ("xc/eta", "beta/eta", "xc^2/epsilon", "xc"))
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = xc ** 2 / xc2_eps
    raw = tab["ExtH"][col] if "ExtH" in tab else math.nan
    ext_h = raw if ExtH and math.isfinite(raw) else None
    unit = auto_unit(col) if ext_h is not None else None
    # exp(-ExtH) per auto unit (the unit of the fit), as a rate per day
    h_ext = math.exp(-ext_h) / UNITS[unit] if ext_h is not None else 0.0
    params = SRParams(eta=eta / xc, beta=beta / xc, epsilon=epsilon / xc ** 2, kappa=0.5 / xc, xc=1.0, h_ext=h_ext,
                      time_unit="days")
    man = json.loads((_dir() / "MANIFEST.json").read_text())
    return params, PresetInfo(name=col, requested=name, type=type, source=source, ExtH=ExtH, ext_h=ext_h,
                              ext_h_unit=unit, kappa_rule=KAPPA_RULE, time_unit="days",
                              file_sha256=man["files"][FILES[type]]["sha256"])
