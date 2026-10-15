"""Asynchronous Background Weather Update Worker.

Periodically queries NOAA HRRR AWS Open Data to ingest and cache newly published forecast cycles.
Features:
- Configurable polling interval and bounded exponential retry backoff.
- Duplicate cycle detection: skips already processed and cached cycles.
- Forecast freshness tracking: monitors age of newest ingested cycle.
- Graceful shutdown via threading.Event or asyncio cancellation.
- Recovery after temporary network outages.
- Persistent metadata ledger recording successfully processed cycles.
- Runs independently of FastAPI request loop to prevent API blocking.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import os
import sys
import threading
import time
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.weather.providers.base import WeatherProviderBase
from src.weather.providers.noaa_hrrr import NOAAHRRRWeatherProvider, DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH: Path = DEFAULT_CACHE_DIR / "processed_cycles.json"
DEFAULT_LOCK_PATH: Path = DEFAULT_CACHE_DIR / "weather_worker.lock"


class WeatherUpdateWorker:
    """Background worker daemon for continuous NOAA HRRR forecast cycle synchronization."""

    def __init__(
        self,
        provider: WeatherProviderBase | None = None,
        poll_interval_seconds: float = 300.0,
        max_retries_per_cycle: int = 3,
        backoff_factor: float = 2.0,
        ledger_path: Path | str = DEFAULT_LEDGER_PATH,
        stale_threshold_hours: float = 6.0,
        lock_path: Path | str | None = None,
        use_process_lock: bool = False,
    ) -> None:
        """Initialize worker.

        Args:
            provider: Weather provider instance. Defaults to NOAAHRRRWeatherProvider.
            poll_interval_seconds: Interval between polling attempts (seconds). Defaults to 300s (5 min).
            max_retries_per_cycle: Max retry attempts when processing an individual cycle.
            backoff_factor: Multiplier for exponential backoff on retry.
            ledger_path: Path to JSON ledger tracking processed cycles.
            stale_threshold_hours: Maximum allowable age of newest cycle before alerting.
            lock_path: Path to singleton lock file.
            use_process_lock: Whether to enforce OS-level cross-process singleton lock.
        """
        self.provider = provider or NOAAHRRRWeatherProvider()
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_retries_per_cycle = int(max_retries_per_cycle)
        self.backoff_factor = float(backoff_factor)
        self.ledger_path = Path(ledger_path)
        self.stale_threshold_hours = float(stale_threshold_hours)
        self.lock_path = Path(lock_path or DEFAULT_LOCK_PATH)
        self.use_process_lock = bool(use_process_lock)
        self._lock_file = None

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.processed_cycles: dict[str, dict[str, Any]] = self._load_ledger()
        self.last_poll_timestamp: str | None = None
        self.last_successful_ingestion: str | None = None
        self.cycle_count: int = len(self.processed_cycles)
        self.is_stale: bool = False
        self.forecast_age_hours: float = 0.0

    def _load_ledger(self) -> dict[str, dict[str, Any]]:
        """Load persistent ledger from disk."""
        if self.ledger_path.is_file():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Could not load ledger from {self.ledger_path}: {exc}")
        return {}

    def _save_ledger(self) -> None:
        """Persist processed cycles ledger to disk."""
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.processed_cycles, f, indent=2)
        except Exception as exc:
            logger.error(f"Failed to persist ledger to {self.ledger_path}: {exc}")

    def sync_once(self) -> dict[str, Any]:
        """Perform a single discovery and download synchronization iteration.

        Returns:
            dict[str, Any]: Iteration summary report.
        """
        self.last_poll_timestamp = datetime.now(timezone.utc).isoformat()
        report: dict[str, Any] = {
            "poll_timestamp_utc": self.last_poll_timestamp,
            "new_cycles_found": 0,
            "cycles_skipped_duplicate": 0,
            "failed_cycles": 0,
            "errors": [],
        }

        try:
            available_cycles = self.provider.discover_latest_cycles(limit=4)
        except Exception as exc:
            msg = f"Failed to discover latest cycles from provider: {exc}"
            logger.warning(msg)
            report["errors"].append(msg)
            return report

        for cycle_meta in available_cycles:
            cycle_str = cycle_meta["cycle_str"]
            if cycle_str in self.processed_cycles:
                report["cycles_skipped_duplicate"] += 1
                continue

            # Process newly discovered cycle across leads (e.g. f01, f02)
            c_date = cycle_meta["cycle_date"]
            c_hour = cycle_meta["cycle_hour"]
            success = False

            if isinstance(self.provider, NOAAHRRRWeatherProvider):
                for attempt in range(self.max_retries_per_cycle):
                    try:
                        field_data = self.provider.get_cycle_field(
                            cycle_date=c_date,
                            cycle_hour=c_hour,
                            lead_hour=2,
                        )
                        self.processed_cycles[cycle_str] = {
                            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "init_time_utc": cycle_meta["init_time_utc"],
                            "valid_time_utc": field_data["valid_time_utc"],
                            "station_temps_f": field_data["station_temps_f"],
                        }
                        report["new_cycles_found"] += 1
                        self.last_successful_ingestion = self.last_poll_timestamp
                        success = True
                        break
                    except Exception as exc:
                        logger.warning(
                            f"Error ingesting cycle {cycle_str} (attempt {attempt + 1}): {exc}"
                        )
                        time.sleep(0.5 * (self.backoff_factor ** attempt))

                if not success:
                    report["failed_cycles"] += 1
                    report["errors"].append(f"Failed to ingest cycle {cycle_str}")
            else:
                self.processed_cycles[cycle_str] = {
                    "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "init_time_utc": cycle_meta.get("init_time_utc"),
                }
                report["new_cycles_found"] += 1

        self._save_ledger()
        self.cycle_count = len(self.processed_cycles)
        self._update_freshness()
        report["total_processed_cycles"] = self.cycle_count
        report["is_stale"] = self.is_stale
        report["forecast_age_hours"] = self.forecast_age_hours

        return report

    def _update_freshness(self) -> None:
        """Calculate age of newest ingested forecast cycle."""
        if not self.processed_cycles:
            self.is_stale = True
            self.forecast_age_hours = 999.0
            return

        latest_init_str: str | None = None
        for cycle_info in self.processed_cycles.values():
            init_ts = cycle_info.get("init_time_utc")
            if init_ts and (latest_init_str is None or init_ts > latest_init_str):
                latest_init_str = init_ts

        if latest_init_str:
            now_utc = pd.Timestamp.now(tz="UTC")
            latest_ts = pd.to_datetime(latest_init_str)
            age_hrs = (now_utc - latest_ts).total_seconds() / 3600.0
            self.forecast_age_hours = max(0.0, round(age_hrs, 2))
            self.is_stale = bool(self.forecast_age_hours > self.stale_threshold_hours)
        else:
            self.is_stale = True
            self.forecast_age_hours = 999.0

    def acquire_lock(self) -> bool:
        """Acquire process-level lock to ensure singleton worker execution."""
        if not self.use_process_lock:
            return True
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(self.lock_path, "a+b")
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            logger.warning(
                f"Singleton worker lock {self.lock_path} is held by another process. Skipping duplicate worker."
            )
            if self._lock_file is not None:
                try:
                    self._lock_file.close()
                except Exception:
                    pass
                self._lock_file = None
            return False

    def release_lock(self) -> None:
        """Release process-level lock if held."""
        if self._lock_file is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self._lock_file.seek(0)
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            finally:
                self._lock_file = None

    def start(self) -> bool:
        """Start the background worker thread.

        Returns:
            bool: True if started successfully, False if process lock could not be acquired.
        """
        if self.use_process_lock and not self.acquire_lock():
            return False

        if self._thread is not None and self._thread.is_alive():
            logger.warning("WeatherUpdateWorker thread is already running.")
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="WeatherUpdateWorkerThread", daemon=True)
        self._thread.start()
        logger.info(f"WeatherUpdateWorker started with interval {self.poll_interval_seconds}s.")
        return True

    def stop(self, timeout_sec: float = 5.0) -> None:
        """Signal the background worker thread to terminate cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
            logger.info("WeatherUpdateWorker stopped cleanly.")
        self.release_lock()

    def is_alive(self) -> bool:
        """Check if worker thread is actively running."""
        return bool(self._thread and self._thread.is_alive())

    def run_forever(self) -> None:
        """Run continuous synchronization loop in foreground until interrupted."""
        if self.use_process_lock and not self.acquire_lock():
            logger.warning("Could not acquire worker singleton lock. Exiting standalone worker.")
            return

        logger.info(f"WeatherUpdateWorker running in foreground (poll={self.poll_interval_seconds}s)...")
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping WeatherUpdateWorker...")
        finally:
            self.stop()

    def _run_loop(self) -> None:
        """Continuous execution loop with interruptible sleep."""
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as exc:
                logger.error(f"Unexpected error in WeatherUpdateWorker loop: {exc}")

            # Sleep in 1-second increments so stop() responds immediately
            for _ in range(int(self.poll_interval_seconds)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker = WeatherUpdateWorker(use_process_lock=True)
    worker.run_forever()

