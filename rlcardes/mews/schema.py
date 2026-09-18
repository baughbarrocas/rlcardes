from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

Scalar = str | int | float


class Site(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class Region(str, Enum):
    REGION_I = "I"
    REGION_II = "II"
    REGION_III = "III"


class Disease(str, Enum):
    LBCL = "LBCL"
    BALL = "B-ALL"
    MM = "MM"


class Target(str, Enum):
    CD19 = "CD19"
    BCMA = "BCMA"


class Costimulation(str, Enum):
    CD28 = "CD28z"
    BB1 = "4-1BBz 1x"
    BB2 = "4-1BBz 2x"


class AgeBand(str, Enum):
    CHILD = "<18"
    ADULT = "18-64"
    OLDER = ">=65"


class Outcome(str, Enum):
    CR = "CR"
    PR = "PR"
    SD = "SD"
    PD = "PD"


@dataclass(frozen=True)
class ClinicalRecord:
    site: Site
    region: Region
    month_index: int
    age_band: AgeBand
    sex: str
    disease: Disease
    target: Target
    costimulation: Costimulation
    hinge: str
    prior_lines: int
    burden: str
    lymphodepletion: str
    exhaustion_flow: float
    exhaustion_polyfunction: float
    persistence_day60: int
    response_day90: Outcome
    pfs_month12: int
    crs_grade: int
    icans_grade: int
    scrna_available: int

    def validate(self) -> None:
        bounded(self.exhaustion_flow, "exhaustion_flow")
        bounded(self.exhaustion_polyfunction, "exhaustion_polyfunction")
        binary(self.persistence_day60, "persistence_day60")
        binary(self.pfs_month12, "pfs_month12")
        binary(self.scrna_available, "scrna_available")
        integer_range(self.month_index, 0, 10_000, "month_index")
        integer_range(self.prior_lines, 0, 9, "prior_lines")
        integer_range(self.crs_grade, 0, 4, "crs_grade")
        integer_range(self.icans_grade, 0, 4, "icans_grade")
        if self.sex not in {"F", "M"}:
            raise ValueError("sex")
        if self.hinge not in {"IgG4", "CD8alpha", "CD28"}:
            raise ValueError("hinge")
        if self.burden not in {"low", "high"}:
            raise ValueError("burden")
        if self.lymphodepletion not in {"Flu/Cy", "other"}:
            raise ValueError("lymphodepletion")
        if self.disease is Disease.MM and self.target is not Target.BCMA:
            raise ValueError("disease_target")
        if self.disease is not Disease.MM and self.target is not Target.CD19:
            raise ValueError("disease_target")


def bounded(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(name)


def binary(value: int, name: str) -> None:
    if value not in {0, 1}:
        raise ValueError(name)


def integer_range(value: int, lower: int, upper: int, name: str) -> None:
    if value < lower or value > upper:
        raise ValueError(name)


@dataclass(frozen=True)
class Profile:
    site: Site
    disease: Disease
    target: Target
    costimulation: Costimulation


PERMITTED_PROFILES: tuple[Profile, ...] = (
    Profile(
        site=Site.A,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.A,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.A,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.A,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.A,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.A,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.A,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.A,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.A,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.B,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.B,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.B,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.B,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.B,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.B,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.B,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.B,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.B,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.C,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.C,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.C,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.C,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.C,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.C,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.C,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.C,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.C,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.D,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.D,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.D,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.D,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.D,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.D,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.D,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.D,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.D,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.E,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.E,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.E,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.E,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.E,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.E,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.E,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.E,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.E,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.F,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.F,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.F,
        disease=Disease.LBCL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.F,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.F,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.F,
        disease=Disease.BALL,
        target=Target.CD19,
        costimulation=Costimulation.BB2,
    ),
    Profile(
        site=Site.F,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.CD28,
    ),
    Profile(
        site=Site.F,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB1,
    ),
    Profile(
        site=Site.F,
        disease=Disease.MM,
        target=Target.BCMA,
        costimulation=Costimulation.BB2,
    ),
)


def from_mapping(row: Mapping[str, Scalar]) -> ClinicalRecord:
    record = ClinicalRecord(
        site=Site(str(row["site_code"])),
        region=Region(str(row["region_code"])),
        month_index=int(row["month_index"]),
        age_band=AgeBand(str(row["age_band"])),
        sex=str(row["sex"]),
        disease=Disease(str(row["disease"])),
        target=Target(str(row["target"])),
        costimulation=Costimulation(str(row["costim_class"])),
        hinge=str(row["hinge"]),
        prior_lines=int(row["prior_lines"]),
        burden=str(row["disease_burden"]),
        lymphodepletion=str(row["lymphodepletion"]),
        exhaustion_flow=float(row["exh_flow"]),
        exhaustion_polyfunction=float(row["exh_polyfunc"]),
        persistence_day60=int(row["persist_d60"]),
        response_day90=Outcome(str(row["response_d90"])),
        pfs_month12=int(row["pfs_12m"]),
        crs_grade=int(row["crs_grade"]),
        icans_grade=int(row["icans_grade"]),
        scrna_available=int(row["scrna_available"]),
    )
    record.validate()
    return record


def external_label(record: ClinicalRecord) -> float:
    return 0.5 * (record.exhaustion_flow + record.exhaustion_polyfunction)


def architecture_concordant(record: ClinicalRecord, threshold: float = 0.41) -> bool:
    expected = Costimulation.CD28 if external_label(record) < threshold else Costimulation.BB2
    return record.costimulation is expected


def severe_crs(record: ClinicalRecord) -> bool:
    return record.crs_grade >= 3


def severe_icans(record: ClinicalRecord) -> bool:
    return record.icans_grade >= 3


def group_by_site(records: list[ClinicalRecord]) -> dict[Site, list[ClinicalRecord]]:
    groups: dict[Site, list[ClinicalRecord]] = {site: [] for site in Site}
    for record in records:
        groups[record.site].append(record)
    return groups


def molecular_subset(records: list[ClinicalRecord]) -> list[ClinicalRecord]:
    return [record for record in records if record.scrna_available == 1]


def profile_supported(record: ClinicalRecord) -> bool:
    candidate = Profile(record.site, record.disease, record.target, record.costimulation)
    return candidate in PERMITTED_PROFILES
