import dataclasses
from collections.abc import Sequence

import gin


@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    input_dim: int
    hidden: tuple[int, ...]
    latent_dim: int
    dropout: float
    n_subpops: int


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    state_dim: int
    hidden: tuple[int, ...]
    n_actions: int
    construct_dim: int


@dataclasses.dataclass(frozen=True)
class PersistSpec:
    phenotype_dim: int
    construct_dim: int
    key_dim: int
    hidden: tuple[int, ...]
    dropout: float


@dataclasses.dataclass(frozen=True)
class SimulatorSpec:
    dt: float
    horizon_days: float
    rho: float
    t_max: float
    delta_a: float
    k_exh: float
    k1: float
    k2: float
    delta_pe: float
    delta_ee: float
    delta_te: float
    growth: float
    b_max: float
    kappa: float
    eta_pe: float


@dataclasses.dataclass(frozen=True)
class PPOSpec:
    clip: float
    gamma: float
    gae_lambda: float
    rollout: int
    n_envs: int
    update_epochs: int
    minibatch: int
    value_coef: float
    entropy_coef: float
    max_grad_norm: float


@dataclasses.dataclass(frozen=True)
class OptimSpec:
    lr: float
    weight_decay: float
    warmup: int
    epochs: int
    batch_size: int
    grad_accum: int
    world_size: int
    patience: int
    precision: str
    beta: float
    reward_weights: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Experiment:
    seed: int
    encoder: EncoderSpec
    policy: PolicySpec
    persist: PersistSpec
    simulator: SimulatorSpec
    ppo: PPOSpec
    optim: OptimSpec

    @property
    def effective_batch(self) -> int:
        return self.optim.batch_size * self.optim.grad_accum * self.optim.world_size


@gin.configurable
def experiment(
    seed: int = 42,
    input_dim: int = 3000,
    encoder_hidden: Sequence[int] = (512, 256, 128),
    latent_dim: int = 128,
    encoder_dropout: float = 0.2,
    n_subpops: int = 8,
    n_actions: int = 45,
    construct_dim: int = 64,
    policy_hidden: Sequence[int] = (256, 128),
    persist_hidden: Sequence[int] = (128, 64),
    key_dim: int = 64,
    persist_dropout: float = 0.3,
    dt: float = 0.1,
    horizon_days: float = 60.0,
    rho: float = 0.5,
    t_max: float = 1e10,
    delta_a: float = 0.02,
    k_exh: float = 0.08,
    k1: float = 0.05,
    k2: float = 0.03,
    delta_pe: float = 0.01,
    delta_ee: float = 0.03,
    delta_te: float = 0.05,
    growth: float = 0.18,
    b_max: float = 1e12,
    kappa: float = 1.2e-9,
    eta_pe: float = 0.4,
    clip: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    rollout: int = 2048,
    n_envs: int = 8,
    update_epochs: int = 4,
    minibatch: int = 256,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    lr: float = 3e-6,
    weight_decay: float = 0.0,
    warmup: int = 200,
    epochs: int = 6000,
    batch_size: int = 1024,
    grad_accum: int = 8,
    world_size: int = 8,
    patience: int = 50,
    precision: str = "bf16",
    beta: float = 0.5,
    reward_weights: Sequence[float] = (0.5, 0.4, 0.1),
) -> Experiment:
    return Experiment(
        seed=seed,
        encoder=EncoderSpec(
            input_dim=input_dim,
            hidden=tuple(encoder_hidden),
            latent_dim=latent_dim,
            dropout=encoder_dropout,
            n_subpops=n_subpops,
        ),
        policy=PolicySpec(
            state_dim=latent_dim + construct_dim,
            hidden=tuple(policy_hidden),
            n_actions=n_actions,
            construct_dim=construct_dim,
        ),
        persist=PersistSpec(
            phenotype_dim=latent_dim,
            construct_dim=construct_dim,
            key_dim=key_dim,
            hidden=tuple(persist_hidden),
            dropout=persist_dropout,
        ),
        simulator=SimulatorSpec(
            dt=dt,
            horizon_days=horizon_days,
            rho=rho,
            t_max=t_max,
            delta_a=delta_a,
            k_exh=k_exh,
            k1=k1,
            k2=k2,
            delta_pe=delta_pe,
            delta_ee=delta_ee,
            delta_te=delta_te,
            growth=growth,
            b_max=b_max,
            kappa=kappa,
            eta_pe=eta_pe,
        ),
        ppo=PPOSpec(
            clip=clip,
            gamma=gamma,
            gae_lambda=gae_lambda,
            rollout=rollout,
            n_envs=n_envs,
            update_epochs=update_epochs,
            minibatch=minibatch,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        ),
        optim=OptimSpec(
            lr=lr,
            weight_decay=weight_decay,
            warmup=warmup,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            world_size=world_size,
            patience=patience,
            precision=precision,
            beta=beta,
            reward_weights=(
                float(reward_weights[0]),
                float(reward_weights[1]),
                float(reward_weights[2]),
            ),
        ),
    )


def load_experiment(recipe_path: str) -> Experiment:
    gin.clear_config()
    gin.parse_config_file(recipe_path)
    return experiment()
