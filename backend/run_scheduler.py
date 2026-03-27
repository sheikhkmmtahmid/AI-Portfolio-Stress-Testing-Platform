"""
Live Data Scheduler
===================
Runs automatically in the background. Start once and leave running.

Schedule:
  Every hour   → fetch asset prices + VIX from Yahoo Finance
  Every day    → fetch full daily market data, re-run Phase 7
  Every month  → fetch FRED macro data, re-run Phase 5 + Phase 7

Usage:
  python run_scheduler.py

  # With FRED API key (for monthly macro updates):
  set FRED_API_KEY=your_key_here
  python run_scheduler.py

  Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from services.live_data_fetcher import run_hourly, run_daily, run_monthly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BACKEND_ROOT / "data" / "live" / "scheduler.log",
                            mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ── Job wrappers (catch all exceptions so scheduler never dies) ───────────────

def job_hourly():
    logger.info("=" * 60)
    logger.info("HOURLY JOB starting")
    try:
        run_hourly()
    except Exception as e:
        logger.error(f"HOURLY JOB unhandled error: {e}")
    logger.info("HOURLY JOB done")


def job_daily():
    logger.info("=" * 60)
    logger.info("DAILY JOB starting")
    try:
        run_daily()
    except Exception as e:
        logger.error(f"DAILY JOB unhandled error: {e}")
    logger.info("DAILY JOB done")


def job_monthly():
    logger.info("=" * 60)
    logger.info("MONTHLY JOB starting")
    try:
        run_monthly()
    except Exception as e:
        logger.error(f"MONTHLY JOB unhandled error: {e}")
    logger.info("MONTHLY JOB done")


# ── Schedule ──────────────────────────────────────────────────────────────────

def job_monthly_conditional():
    """Runs daily at 08:00 but only executes the monthly job on the 1st of the month."""
    if datetime.utcnow().day == 1:
        job_monthly()
    else:
        logger.debug("Monthly job skipped — not the 1st of the month.")


def setup_schedule():
    # Hourly — every hour on the hour
    schedule.every().hour.at(":00").do(job_hourly)

    # Daily — every day at 22:00 UTC (after US market close)
    schedule.every().day.at("22:00").do(job_daily)

    # Monthly — checked daily at 08:00 UTC, but only fires on the 1st of the month
    schedule.every().day.at("08:00").do(job_monthly_conditional)

    logger.info("Schedule configured:")
    logger.info("  Hourly  : every hour at :00  — Yahoo Finance prices")
    logger.info("  Daily   : 22:00 UTC           — Yahoo Finance full market data + Phase 7")
    logger.info("  Monthly : 1st of month 08:00  — FRED macro data + Phase 5 + Phase 7")


def run_startup_jobs():
    """On startup, run all jobs immediately to populate fresh data."""
    logger.info("Running startup data fetch (all frequencies)...")
    job_hourly()
    job_daily()
    job_monthly()
    logger.info("Startup fetch complete. Scheduler now running.")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Live Data Scheduler starting")
    logger.info(f"Backend root: {BACKEND_ROOT}")
    logger.info("=" * 60)

    # Run all jobs immediately on startup so data is fresh right away
    run_startup_jobs()

    # Set up recurring schedule
    setup_schedule()

    logger.info("Entering scheduler loop. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            # Log next run times every 15 minutes
            now = datetime.utcnow()
            if now.minute % 15 == 0 and now.second < 5:
                jobs = schedule.get_jobs()
                for job in jobs:
                    logger.info(f"  Next run: {job.next_run}  — {job.job_func.__name__}")
            time.sleep(30)   # check every 30 seconds
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
