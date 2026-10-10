#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base observer classes for monitoring frame flow in the Pipecat pipeline.

This module provides the foundation for observing frame transfers between
processors without modifying the pipeline structure. Observers can be used
for logging, debugging, analytics, and monitoring pipeline behavior.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class FrameProcessed:
    """Event data for frame processing in the pipeline.

    Represents an event where a frame is being processed by a processor. This
    data structure is typically used by observers to track the flow of frames
    through the pipeline for logging, debugging, or analytics purposes.

    Parameters:
        processor: The processor processing the frame.
        frame: The frame being processed.
        direction: The direction of the frame (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.

    """

    processor: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


@dataclass
class FramePushed:
    """Event data for frame transfers between processors in the pipeline.

    Represents an event where a frame is pushed from one processor to another
    within the pipeline. This data structure is typically used by observers
    to track the flow of frames through the pipeline for logging, debugging,
    or analytics purposes.

    Parameters:
        source: The processor sending the frame.
        destination: The processor receiving the frame.
        frame: The frame being transferred.
        direction: The direction of the transfer (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.
    """

    source: "FrameProcessor"
    destination: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.

    Two class attributes let the worker skip work an observer never wanted
    (see ``WorkerObserver``). Every frame hop between processors raises one
    push and one process event, and the caller's raw audio alone is ~50 frames
    a second through every processor — on a phone call that is thousands of
    events a second per observer, each queued to its own task.

    Attributes:
        ignored_frame_types: Frame types this observer never looks at. The
            worker does not hand it events for them. Leave empty (the default)
            to see every frame.
        inline_dispatch: Call this observer directly instead of through its own
            queue and task. Only for observers whose handlers are a few
            compares and stores and never await anything slow: they run on the
            pipeline's own path.
    """

    ignored_frame_types: tuple = ()
    inline_dispatch: bool = False

    async def on_process_frame(self, data: FrameProcessed):
        """Handle the event when a frame is being processed by a processor.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        being processed by a processor.

        Args:
            data: The event data containing details about the frame processing.
        """
        pass

    async def on_push_frame(self, data: FramePushed):
        """Handle the event when a frame is pushed from one processor to another.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        transferred through the pipeline.

        Args:
            data: The event data containing details about the frame transfer.
        """
        pass

    async def on_pipeline_started(self):
        """Called when the pipeline has fully started.

        Fired after the ``StartFrame`` has been processed by all processors
        in the pipeline, including nested ``ParallelPipeline`` branches.
        """
        pass
