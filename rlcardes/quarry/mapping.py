from rlcardes.jesses.config import SimulatorSpec
from rlcardes.mews.library import Construct
from rlcardes.quarry.kinetics import SimParams

COSTIM_RHO: tuple[float, ...] = (1.4, 1.1, 1.25, 1.15, 1.05)
COSTIM_KEXH: tuple[float, ...] = (1.3, 0.7, 1.1, 0.95, 0.85)
HINGE_KAPPA: tuple[float, ...] = (1.0, 0.9, 1.1)
ITAM_COUNT: tuple[int, ...] = (1, 2, 3)


def signaling_strength(construct: Construct) -> float:
    drive = COSTIM_RHO[construct.costim] * ITAM_COUNT[construct.itam] / 4.2
    bounded = 0.3 + 0.7 * drive
    return float(min(max(bounded, 0.3), 1.0))


def map_construct(construct: Construct, base: SimulatorSpec) -> SimParams:
    return SimParams(
        rho=base.rho * COSTIM_RHO[construct.costim] * ITAM_COUNT[construct.itam],
        sigma=signaling_strength(construct),
        k_exh=base.k_exh * COSTIM_KEXH[construct.costim],
        kappa=base.kappa * HINGE_KAPPA[construct.hinge],
        k1=base.k1,
        k2=base.k2,
        delta_a=base.delta_a,
        delta_pe=base.delta_pe,
        delta_ee=base.delta_ee,
        delta_te=base.delta_te,
        growth=base.growth,
        t_max=base.t_max,
        b_max=base.b_max,
        eta_pe=base.eta_pe,
    )
