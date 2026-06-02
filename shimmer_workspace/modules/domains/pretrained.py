from collections.abc import Mapping, Sequence
from pathlib import Path

from shimmer import DomainModule, GWDecoder, GWEncoder
from torch.nn import Linear, Module

from shimmer_metaworld import PROJECT_DIR
from shimmer_metaworld.ckpt_migrations import (
    migrate_model,
)
from shimmer_metaworld.config import DomainModuleVariant, LoadedDomainConfig
from shimmer_metaworld.errors import ConfigurationError
from shimmer_metaworld.modules.domains.attribute import (
    ActionDomainModule,
    AttributeLegacyDomainModule,
    ActionLegacyDomainModule,
)
from shimmer_metaworld.modules.domains.visual import (
    VisualDomainModule,
    VisualLatentDomainModule,
    VisualLatentDomainWithUnpairedModule,
)
from torch import nn

def get_n_layers(n_layers: int, hidden_dim: int) -> list[nn.Module]:
    """
    Makes a list of `n_layers` `nn.Linear` layers with `nn.ReLU`.

    Args:
        n_layers (`int`): number of layers
        hidden_dim (`int`): size of the hidden dimension

    Returns:
        `list[nn.Module]`: list of linear and relu layers.
    """
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class ActDecoder(nn.Sequential):
    """A Decoder network for GWModules."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """
        Initializes the decoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """

        self.in_dim = in_dim
        """input dimension"""

        self.hidden_dim = hidden_dim
        """hidden dimension"""

        self.out_dim = out_dim
        """output dimension"""

        self.n_layers = n_layers
        """
        number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        super().__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.z_dim)


def load_pretrained_module(domain: LoadedDomainConfig) -> DomainModule:
    domain_checkpoint = Path(domain.checkpoint_path)
    module: DomainModule
    match domain.domain_type:
        case DomainModuleVariant.v:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_metaworld" / "migrations" / "visual_mod",
            )
            module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )

        case DomainModuleVariant.v_latents:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_metaworld" / "migrations" / "visual_mod",
            )
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainModule(v_module)

        case DomainModuleVariant.v_latents_unpaired:
            migrate_model(
                domain_checkpoint,
                PROJECT_DIR / "shimmer_metaworld" / "migrations" / "visual_mod",
            )
            v_module = VisualDomainModule.load_from_checkpoint(
                domain_checkpoint, **domain.args
            )
            module = VisualLatentDomainWithUnpairedModule(v_module)

        case DomainModuleVariant.attr:
            module = AttributeLegacyDomainModule()
        
        case DomainModuleVariant.act:
         module = ActionLegacyDomainModule()

        case _:
            raise ConfigurationError(f"Unknown domain type {domain.domain_type.name}")
    return module
'''
    migrate_model(
        domain_checkpoint,
        PROJECT_DIR / "shimmer_metaworld" / "migrations" / "act_mod",
    )
    module = ActionDomainModule.load_from_checkpoint(
        domain_checkpoint, **domain.args
    )
'''
'''
case DomainModuleVariant.attr:
    migrate_model(
        domain_checkpoint,
        PROJECT_DIR / "shimmer_metaworld" / "migrations" / "attr_mod",
    )
    module = AttributeDomainModule.load_from_checkpoint(
        domain_checkpoint, **domain.args
    )
'''

def get_from_dict_or_val(
    val: int | Mapping[DomainModuleVariant, int], key: DomainModuleVariant, log: str
) -> int:
    """
    If val is int, return val, otherwise return val[key]
    """
    if isinstance(val, int):
        return val

    assert key in val, f"{key} should be defined in {log}."
    return val[key]


def load_pretrained_domain(
    domain: LoadedDomainConfig,
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[DomainModuleVariant, int],
    encoders_n_layers: int | Mapping[DomainModuleVariant, int],
    decoders_hidden_dim: int | Mapping[DomainModuleVariant, int],
    decoders_n_layers: int | Mapping[DomainModuleVariant, int],
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[DomainModule, Module, Module]:
    module = load_pretrained_module(domain)

    encoder_hidden_dim = get_from_dict_or_val(
        encoders_hidden_dim, domain.domain_type, "global_workspace.encoders.hidden_dim"
    )
    decoder_hidden_dim = get_from_dict_or_val(
        decoders_hidden_dim, domain.domain_type, "global_workspace.decoders.hidden_dim"
    )

    encoder_n_layers = get_from_dict_or_val(
        encoders_n_layers, domain.domain_type, "global_workspace.encoder.n_layers"
    )

    decoder_n_layers = get_from_dict_or_val(
        decoders_n_layers, domain.domain_type, "global_workspace.decoders.n_layers"
    )

    gw_encoder: Module
    gw_decoder: Module
    if is_linear:
        gw_encoder = Linear(module.latent_dim, workspace_dim, bias=bias)
        gw_decoder = Linear(workspace_dim, module.latent_dim, bias=bias)
    else:
        gw_encoder = GWEncoder(
            module.latent_dim, encoder_hidden_dim, workspace_dim, encoder_n_layers
        )
        gw_decoder = GWDecoder(
            workspace_dim, decoder_hidden_dim, module.latent_dim, decoder_n_layers
        )

    return module, gw_encoder, gw_decoder
'''
 elif isinstance(module, ActionLegacyDomainModule):
        gw_encoder = GWEncoder(
            module.latent_dim, encoder_hidden_dim, workspace_dim, encoder_n_layers
        )
        gw_decoder = ActDecoder(
            workspace_dim, decoder_hidden_dim, module.latent_dim, decoder_n_layers
        )
'''

def load_pretrained_domains(
    domains: Sequence[LoadedDomainConfig],
    workspace_dim: int,
    encoders_hidden_dim: int | Mapping[DomainModuleVariant, int],
    encoders_n_layers: int | Mapping[DomainModuleVariant, int],
    decoders_hidden_dim: int | Mapping[DomainModuleVariant, int],
    decoders_n_layers: int | Mapping[DomainModuleVariant, int],
    is_linear: bool = False,
    bias: bool = False,
) -> tuple[dict[str, DomainModule], dict[str, Module], dict[str, Module]]:
    modules: dict[str, DomainModule] = {}
    gw_encoders: dict[str, Module] = {}
    gw_decoders: dict[str, Module] = {}
    for domain in domains:
        if domain.domain_type.kind.value.kind in modules:
            raise ConfigurationError("Cannot load multiple domains of the same kind.")
        model, encoder, decoder = load_pretrained_domain(
            domain,
            workspace_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
            is_linear,
            bias,
        )
        modules[domain.domain_type.kind.value.kind] = model
        gw_encoders[domain.domain_type.kind.value.kind] = encoder
        gw_decoders[domain.domain_type.kind.value.kind] = decoder
    return modules, gw_encoders, gw_decoders
