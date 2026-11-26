"""Adapter over Prefect's built-in REST API client (``prefect.client.orchestration.PrefectClient``).

The adapter receives the API URL and optional API key from configuration (``PREFECT_API_URL``,
``PREFECT_API_KEY``) and returns only non-secret application details. Each detail records the
client operation and underlying Prefect REST endpoint used to retrieve it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from telco_churn_dataops.config import Settings
from telco_churn_dataops.logging_utils import isoformat


class PrefectUnavailableError(RuntimeError):
    """Raised when the Prefect API is not configured or cannot be reached."""

    def __init__(self, message: str, *, reason: str, api_url: str | None):
        super().__init__(message)
        self.reason = reason
        self.api_url = api_url


class PrefectDetailsProvider(Protocol):
    """Interface used by the FastAPI layer (mocked in tests)."""

    async def get_application_details(self) -> dict[str, Any]:
        """Return flow, deployment, flow-run and work-pool details."""
        ...


def _enum_str(value: Any) -> str | None:
    if value is None:
        return None
    val = getattr(value, "value", value)
    return str(val) or None


def _dt(value: Any) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else value


def _schedules(deployment: Any) -> list[dict[str, Any]]:
    out = []
    for item in getattr(deployment, "schedules", None) or []:
        sched = item.schedule
        interval = getattr(sched, "interval", None)
        out.append(
            {
                "id": str(item.id),
                "active": bool(item.active),
                "type": type(sched).__name__,
                "interval_seconds": interval.total_seconds() if interval is not None else None,
                "cron": getattr(sched, "cron", None),
                "anchor_date": _dt(getattr(sched, "anchor_date", None)),
                "timezone": getattr(sched, "timezone", None),
            }
        )
    return out


class PrefectApiAdapter:
    """Retrieve application details from the Prefect server or Prefect Cloud."""

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[str, str | None], Any] | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.settings = settings
        self.timeout_seconds = timeout_seconds
        self._client_factory = client_factory or self._default_client

    def _default_client(self, api_url: str, api_key: str | None) -> Any:
        from prefect.client.orchestration import PrefectClient

        return PrefectClient(
            api_url, api_key=api_key, httpx_settings={"timeout": self.timeout_seconds}
        )

    async def get_application_details(self) -> dict[str, Any]:
        """Query flow, deployment, recent flow runs and work pool/workers via the Prefect API."""
        api_url = self.settings.prefect_api_url
        if not api_url:
            raise PrefectUnavailableError(
                "PREFECT_API_URL is not configured", reason="not_configured", api_url=None
            )
        key = self.settings.prefect_api_key
        client = self._client_factory(api_url, key.get_secret_value() if key else None)
        flow_name = self.settings.prefect_flow_name
        dep_name = self.settings.prefect_deployment_name
        try:
            async with client:
                health_error = await client.api_healthcheck()
                if health_error is not None:
                    raise PrefectUnavailableError(
                        f"Prefect API health check failed: {type(health_error).__name__}",
                        reason="unreachable",
                        api_url=api_url,
                    )
                deployment = await client.read_deployment_by_name(f"{flow_name}/{dep_name}")
                flow = await client.read_flow(deployment.flow_id)
                runs = await self._recent_runs(client, deployment.id)
                upcoming = await self._next_scheduled_runs(client, deployment.id)
                pool_name = deployment.work_pool_name or self.settings.prefect_work_pool
                pool = await client.read_work_pool(pool_name)
                workers = await client.read_workers_for_work_pool(pool_name, limit=10)
        except PrefectUnavailableError:
            raise
        except Exception as exc:
            reason = "not_found" if type(exc).__name__ == "ObjectNotFound" else "request_failed"
            raise PrefectUnavailableError(
                f"Prefect API request failed: {type(exc).__name__}: {str(exc)[:200]}",
                reason=reason,
                api_url=api_url,
            ) from exc

        started = [r for r in runs if r.start_time is not None]
        latest = started[0] if started else (runs[0] if runs else None)
        scheduled = [r for r in upcoming if r.start_time is None and r.state_name == "Scheduled"]
        return {
            "retrieved_at": isoformat(),
            "prefect_api": {"url": api_url, "is_cloud": "api.prefect.cloud" in api_url},
            "flow": {
                "source": "PrefectClient.read_flow -> GET /flows/{id}",
                "id": str(flow.id),
                "name": flow.name,
                "created": _dt(flow.created),
                "tags": list(flow.tags or []),
            },
            "deployment": {
                "source": "PrefectClient.read_deployment_by_name -> "
                "GET /deployments/name/{flow_name}/{deployment_name}",
                "id": str(deployment.id),
                "name": deployment.name,
                "entrypoint": deployment.entrypoint,
                "paused": bool(getattr(deployment, "paused", False)),
                "status": _enum_str(getattr(deployment, "status", None)),
                "concurrency_limit": (
                    getattr(deployment, "concurrency_limit", None)
                    or getattr(getattr(deployment, "global_concurrency_limit", None), "limit", None)
                ),
                "work_pool_name": deployment.work_pool_name,
                "work_queue_name": deployment.work_queue_name,
                "schedules": _schedules(deployment),
                "updated": _dt(deployment.updated),
            },
            "latest_flow_run": {
                "source": "PrefectClient.read_flow_runs -> POST /flow_runs/filter",
                "id": str(latest.id) if latest else None,
                "name": latest.name if latest else None,
                "state_type": _enum_str(latest.state_type) if latest else None,
                "state_name": latest.state_name if latest else None,
                "expected_start_time": _dt(latest.expected_start_time) if latest else None,
                "start_time": _dt(latest.start_time) if latest else None,
                "end_time": _dt(latest.end_time) if latest else None,
                "total_run_time_seconds": latest.total_run_time.total_seconds() if latest else None,
            },
            "next_scheduled_run": (
                {
                    "name": scheduled[0].name,
                    "expected_start_time": _dt(scheduled[0].expected_start_time),
                }
                if scheduled
                else None
            ),
            "recent_flow_runs": [
                {
                    "name": r.name,
                    "state_name": r.state_name,
                    "start_time": _dt(r.start_time),
                    "end_time": _dt(r.end_time),
                }
                for r in runs
            ],
            "work_pool": {
                "source": "PrefectClient.read_work_pool -> GET /work_pools/{name}; "
                "read_workers_for_work_pool -> POST /work_pools/{name}/workers/filter",
                "id": str(pool.id),
                "name": pool.name,
                "type": pool.type,
                "status": _enum_str(getattr(pool, "status", None)),
                "is_paused": bool(pool.is_paused),
                "concurrency_limit": pool.concurrency_limit,
                "workers": [
                    {
                        "name": w.name,
                        "status": _enum_str(getattr(w, "status", None)),
                        "last_heartbeat_time": _dt(w.last_heartbeat_time),
                    }
                    for w in workers
                ],
            },
        }

    async def _recent_runs(self, client: Any, deployment_id: Any, limit: int = 5) -> list[Any]:
        from prefect.client.schemas.filters import (
            DeploymentFilter,
            DeploymentFilterId,
            FlowRunFilter,
            FlowRunFilterStartTime,
        )
        from prefect.client.schemas.sorting import FlowRunSort

        fetch: Callable[..., Awaitable[list[Any]]] = client.read_flow_runs
        return await fetch(
            deployment_filter=DeploymentFilter(id=DeploymentFilterId(any_=[deployment_id])),
            flow_run_filter=FlowRunFilter(start_time=FlowRunFilterStartTime(is_null_=False)),
            sort=FlowRunSort.START_TIME_DESC,
            limit=limit,
        )

    async def _next_scheduled_runs(
        self, client: Any, deployment_id: Any, limit: int = 3
    ) -> list[Any]:
        from prefect.client.schemas.filters import (
            DeploymentFilter,
            DeploymentFilterId,
            FlowRunFilter,
            FlowRunFilterStartTime,
        )
        from prefect.client.schemas.sorting import FlowRunSort

        fetch: Callable[..., Awaitable[list[Any]]] = client.read_flow_runs
        return await fetch(
            deployment_filter=DeploymentFilter(id=DeploymentFilterId(any_=[deployment_id])),
            flow_run_filter=FlowRunFilter(start_time=FlowRunFilterStartTime(is_null_=True)),
            sort=FlowRunSort.EXPECTED_START_TIME_ASC,
            limit=limit,
        )
