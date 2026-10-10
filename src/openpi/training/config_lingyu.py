"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.teleavatar_v2_policy as teleavatar_v2_policy
import openpi.policies.teleavatar_v2_policy_norm as teleavatar_v2_policy_norm
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Lingyu Dataset V2
    iceberg_dir: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # An absolute-path repo_id (local dataset) must not be used as the
        # asset_id: every `... / asset_id` join would resolve to the dataset
        # directory (absolute paths win in path joins), so checkpoints's
        # assets/ would silently stay empty and serving would only work with
        # the dataset mounted at the same path. Use the dataset basename
        # instead, so norm stats really get packaged into the checkpoint.
        is_local_path = repo_id is not None and pathlib.PurePath(repo_id).is_absolute()
        asset_id = self.assets.asset_id or (pathlib.PurePath(repo_id).name if is_local_path else repo_id)
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        if norm_stats is None and is_local_path:
            # Convention for local datasets: compute_norm_stats.py writes
            # <dataset>/norm_stats.json (its output path joins the absolute
            # repo_id). Load from there so the stats live with the data and
            # get copied into the checkpoint's assets/<basename>/ on save.
            norm_stats = self._load_norm_stats(epath.Path(repo_id).parent, pathlib.PurePath(repo_id).name)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=norm_stats,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTeleavatarV2DataConfig(DataConfigFactory):
    """
    Config for training on the Teleavatar v2 dual-arm robot dataset.

    Handles the v2 proprioceptive state (62-dim: positions, velocities,
    efforts + EE poses; 72-dim datasets append chassis dims — the indices
    used are identical) and 3 side-by-side stereo camera feeds:
    TeleavatarInputs crops one eye per camera (head → left eye,
    left → right eye, right → left eye). For the v1 (officially released)
    robot use LeRobotTeleavatarV1DataConfig instead.
    """
    use_delta_joint_actions: bool = False
    # Whether the head camera should be rotated 180° before the left-eye crop.
    # Property of the source dataset orientation; forwarded to TeleavatarInputs.
    rotate_head_camera: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform to match dataset keys to inference keys.
        repack_structure = {
            "observation/images/left_color": "observation.images.left_color",
            "observation/images/right_color": "observation.images.right_color",
            "observation/images/head_camera": "observation.images.head_camera",
            "observation/state": "observation.state",
            "action": "action",  # Keep action as action
        }
        # When prompt_from_task is on, PromptFromLeRobotTask injects a top-level
        # "prompt" string from meta.tasks[task_index]. RepackTransform rebuilds
        # the dict from scratch, so the key has to be listed here or it's lost
        # and TeleavatarInputs falls back to its hardcoded default for every
        # sample. Only request the key when it will actually be present;
        # otherwise the flat_item lookup would KeyError.
        base_cfg = self.base_config or DataConfig()
        if base_cfg.prompt_from_task:
            repack_structure["prompt"] = "prompt"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        # Delta is handled inside TeleavatarInputs/Outputs (see there for why).
        data_transforms = _transforms.Group(
            inputs=[
                teleavatar_v2_policy.TeleavatarInputs(
                    model_type=model_config.model_type,
                    rotate_head_camera=self.rotate_head_camera,
                    use_delta_joint_actions=self.use_delta_joint_actions,
                )
            ],
            outputs=[
                teleavatar_v2_policy.TeleavatarOutputs(
                    use_delta_joint_actions=self.use_delta_joint_actions,
                )
            ],
        )

        # Model transforms
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LingyuTeleavatarV2DataConfig(DataConfigFactory):
    """
    Config for training on the Teleavatar v2 dual-arm robot dataset.
    """
    # 灵御数据集不来自 LeRobot, 用不到 repo_id; 给默认值后 tyro 不再要求命令行必须传 --data.repo-id
    repo_id: str | None = None
    use_delta_joint_actions: bool = False
    # Whether the head camera should be rotated 180° before the left-eye crop.
    # Property of the source dataset orientation; forwarded to TeleavatarInputs.
    rotate_head_camera: bool = False

    compute_norm_stats: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform to match dataset keys to inference keys.
        if not self.compute_norm_stats:
            repack_structure = {
                # 源 key 用 "/" 分隔: RepackTransform 按 "/" 展平 LingyuDatasetV2 输出的嵌套 dict
                "observation/images/left_color": "observation/images/left_color",
                "observation/images/right_color": "observation/images/right_color",
                "observation/images/head_camera": "observation/images/head_camera",
                "observation/state": "observation/state",
                "action": "action",  # Keep action as action
            }
        else:
            # 只算 norm stats: 数据集不读图像, 故只 repack state/action
            repack_structure = {
                "observation/state": "observation/state",
                "action": "action",  # Keep action as action
            }
        # When prompt_from_task is on, PromptFromLeRobotTask injects a top-level
        # "prompt" string from meta.tasks[task_index]. RepackTransform rebuilds
        # the dict from scratch, so the key has to be listed here or it's lost
        # and TeleavatarInputs falls back to its hardcoded default for every
        # sample. Only request the key when it will actually be present;
        # otherwise the flat_item lookup would KeyError.
        base_cfg = self.base_config or DataConfig()
        if base_cfg.prompt_from_task:
            repack_structure["prompt"] = "prompt"
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_structure)
            ]
        )

        # Delta is handled inside TeleavatarInputs/Outputs (see there for why).
        # 算 norm stats 时用不处理图像的 TeleavatarInputs, state/action 的处理与训练一致
        inputs_policy = teleavatar_v2_policy_norm if self.compute_norm_stats else teleavatar_v2_policy
        data_transforms = _transforms.Group(
            inputs=[
                inputs_policy.TeleavatarInputs(
                    model_type=model_config.model_type,
                    rotate_head_camera=self.rotate_head_camera,
                    use_delta_joint_actions=self.use_delta_joint_actions,
                    state_action_selected=True,
                )
            ],
            outputs=[
                teleavatar_v2_policy.TeleavatarOutputs(
                    use_delta_joint_actions=self.use_delta_joint_actions,
                )
            ],
        )

        # Model transforms
        model_transforms = ModelTransformFactory()(model_config)

        base_data_config = self.create_base_config(assets_dirs, model_config)
        if base_data_config.asset_id is None:
            # 没有 repo_id 时 compute_norm_stats_lingyu.py 把 norm stats 直接写在 assets_dirs 下: 以其目录名作 asset_id
            # 从那里读; checkpoint 保存与推理加载 norm stats 也都按 asset_id 找, 不能留空
            base_data_config = dataclasses.replace(
                base_data_config, asset_id=assets_dirs.name,
                norm_stats=self._load_norm_stats(epath.Path(assets_dirs).parent, assets_dirs.name))

        return dataclasses.replace(
            base_data_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 32
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Fine-tuning Teleavatar v2 configs.
    #
    TrainConfig(
        name="pi0_teleavatar_v2",
        # Full fine-tune of pi0 on Teleavatar V2 data.
        model=pi0_config.Pi0Config(
            action_dim=32,  # Keep 32 to match pi0_base pretrained weights
            action_horizon=30
        ),
        data=LeRobotTeleavatarV2DataConfig(
            repo_id="MK_tower_floor2_lerobot_45fps",  # Your local dataset name
            base_config=DataConfig(
                prompt_from_task=True,  # Read the language instruction from the LeRobot task field
                action_sequence_keys=("action",)  # Use 'action' not 'actions'
            ),
            use_delta_joint_actions=False,  # Absolute joint positions (not deltas)
            # v2 robot: head camera is right-side-up, so no 180° rotation
            # before the left-eye crop. Set True only for v1 datasets, whose
            # head camera was mounted upside-down.
            rotate_head_camera=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        batch_size=64,
        num_train_steps=20000,

    ),
    TrainConfig(
        name="pi0_teleavatar_v2_lingyu",
        model=pi0_config.Pi0Config(
            action_dim=32,  # Keep 32 to match pi0_base pretrained weights
            action_horizon=30,
        ),
        checkpoint_base_dir="./checkpoints",
        data=LingyuTeleavatarV2DataConfig(
            base_config=DataConfig(
                action_sequence_keys=("action",),  # Use 'action' not 'actions'
                prompt_from_task=True,
                iceberg_dir="/mnt/dex/a800/home/shihaoran/openpi/src/openpi/training/lingyu_dataloader_v2/iceberg_warehouse",
            ),
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/data/base_model/pi0_base/params"),
        batch_size=64,
        num_workers=256,
        num_train_steps=200_000,
        wandb_enabled=True,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
