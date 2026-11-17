"""LeRobot processor format; SigLIP and state normalization happen inside policy."""

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_flowmatch_pre_post_processors(config, dataset_stats=None):
    # Values are already native q/gPO space. Do not reuse ACMT crop, residual
    # restoration, gripper mapping, or dataset-derived normalization processors.
    del dataset_stats
    return (
        PolicyProcessorPipeline(
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
