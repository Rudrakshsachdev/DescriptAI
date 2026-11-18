"""Process-local controlled concurrency for synchronous model inference."""

import asyncio
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from backend.app.core.exceptions import (
    InferenceTimeoutException,
    ServiceOverloadedException,
)
from backend.app.core.logging import get_logger

logger = get_logger("backend.concurrency")
T = TypeVar("T")


class InferenceConcurrencyLimiter:
    """Bounds synchronous inference and keeps the asyncio event loop responsive."""

    def __init__(
        self,
        max_concurrent: int,
        capacity_wait_seconds: float,
        inference_timeout_seconds: float,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if capacity_wait_seconds < 0:
            raise ValueError("capacity_wait_seconds cannot be negative")
        if inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be positive")

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.capacity_wait_seconds = capacity_wait_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self.max_concurrent = max_concurrent
        self._active_tasks: set[asyncio.Task[Any]] = set()

    @property
    def available_slots(self) -> int:
        """Returns the approximate number of currently available inference slots."""
        return max(0, self._semaphore._value)

    async def run(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Runs synchronous inference with bounded capacity and a request timeout."""
        acquisition = asyncio.create_task(self._semaphore.acquire())
        try:
            completed, _ = await asyncio.wait(
                {acquisition},
                timeout=self.capacity_wait_seconds,
            )
            if acquisition not in completed:
                self._cancel_acquisition(acquisition)
                logger.warning(
                    "Inference overload: capacity=%d unavailable within %.3f seconds",
                    self.max_concurrent,
                    self.capacity_wait_seconds,
                )
                raise ServiceOverloadedException(
                    "The description generation service is currently overloaded.",
                )
            await acquisition
        except asyncio.CancelledError:
            self._cancel_acquisition(acquisition)
            raise

        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._active_tasks.add(task)
        timed_out = False

        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            timed_out = True
            logger.warning(
                "Inference timeout after %.3f seconds; underlying work may still be running",
                self.inference_timeout_seconds,
            )
            task.add_done_callback(self._release_after_task)
            raise InferenceTimeoutException(
                "The description generation request timed out.",
            ) from exc
        except asyncio.CancelledError:
            timed_out = True
            task.add_done_callback(self._release_after_task)
            raise
        except BaseException:
            raise
        finally:
            if not timed_out:
                if task.done():
                    self._active_tasks.discard(task)
                    if not task.cancelled():
                        task.exception()
                    self._semaphore.release()
                else:
                    task.add_done_callback(self._release_after_task)

    def _cancel_acquisition(self, acquisition: asyncio.Task[bool]) -> None:
        """Cancels a pending semaphore acquire and releases a permit if it won a race."""
        acquisition.cancel()
        acquisition.add_done_callback(self._release_acquired_permit)

    def _release_acquired_permit(self, acquisition: asyncio.Task[bool]) -> None:
        """Returns a permit acquired concurrently with cancellation or overload."""
        if acquisition.cancelled():
            return
        error = acquisition.exception()
        if error is not None:
            logger.error(
                "Semaphore acquisition failed during cancellation cleanup.",
                exc_info=(type(error), error, error.__traceback__),
            )
            return
        if acquisition.result():
            self._semaphore.release()

    def _release_after_task(self, task: asyncio.Future[T]) -> None:
        """Releases capacity only after underlying synchronous inference finishes."""
        if not task.cancelled():
            task.exception()
        self._active_tasks.discard(task)
        self._semaphore.release()

    async def wait_for_idle(self, timeout: Optional[float] = None) -> bool:
        """Waits for active synchronous inference work to finish during shutdown."""
        if not self._active_tasks:
            return True
        pending = tuple(self._active_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False
