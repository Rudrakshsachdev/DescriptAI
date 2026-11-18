"""Reliability tests for bounded inference, overload, and timeout behavior."""

import asyncio
import time

import pytest

from backend.app.core.concurrency import InferenceConcurrencyLimiter
from backend.app.core.exceptions import (
    InferenceTimeoutException,
    ServiceOverloadedException,
)


def test_limiter_releases_capacity_after_success():
    async def scenario() -> None:
        limiter = InferenceConcurrencyLimiter(
            max_concurrent=1,
            capacity_wait_seconds=0.1,
            inference_timeout_seconds=1.0,
        )

        first = await limiter.run(lambda: "success")
        second = await limiter.run(lambda: "again")

        assert first == "success"
        assert second == "again"
        assert limiter.available_slots == 1

    asyncio.run(scenario())


def test_limiter_releases_capacity_after_exception():
    async def scenario() -> None:
        limiter = InferenceConcurrencyLimiter(
            max_concurrent=1,
            capacity_wait_seconds=0.1,
            inference_timeout_seconds=1.0,
        )

        def fail() -> None:
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError):
            await limiter.run(fail)
        assert await limiter.run(lambda: "recovered") == "recovered"

    asyncio.run(scenario())


def test_limiter_reports_overload_when_capacity_is_busy():
    async def scenario() -> None:
        limiter = InferenceConcurrencyLimiter(
            max_concurrent=1,
            capacity_wait_seconds=0.02,
            inference_timeout_seconds=1.0,
        )

        busy = asyncio.create_task(limiter.run(lambda: time.sleep(0.08)))
        await asyncio.sleep(0.01)

        with pytest.raises(ServiceOverloadedException):
            await limiter.run(lambda: "not accepted")

        await busy

    asyncio.run(scenario())


def test_timeout_does_not_release_capacity_while_inference_is_running():
    async def scenario() -> None:
        limiter = InferenceConcurrencyLimiter(
            max_concurrent=1,
            capacity_wait_seconds=0.02,
            inference_timeout_seconds=0.02,
        )

        def slow() -> str:
            time.sleep(0.12)
            return "late result"

        with pytest.raises(InferenceTimeoutException):
            await limiter.run(slow)

        with pytest.raises(ServiceOverloadedException):
            await limiter.run(lambda: "must remain blocked")

        assert await limiter.wait_for_idle(timeout=1.0) is True
        assert await limiter.run(lambda: "released") == "released"

    asyncio.run(scenario())


def test_cancellation_does_not_release_capacity_before_underlying_inference_finishes():
    async def scenario() -> None:
        limiter = InferenceConcurrencyLimiter(
            max_concurrent=1,
            capacity_wait_seconds=0.02,
            inference_timeout_seconds=1.0,
        )

        def slow() -> str:
            time.sleep(0.12)
            return "completed"

        request = asyncio.create_task(limiter.run(slow))
        await asyncio.sleep(0.01)
        request.cancel()

        with pytest.raises(asyncio.CancelledError):
            await request

        with pytest.raises(ServiceOverloadedException):
            await limiter.run(lambda: "must remain blocked")

        assert await limiter.wait_for_idle(timeout=1.0) is True
        assert await limiter.run(lambda: "released") == "released"

    asyncio.run(scenario())
