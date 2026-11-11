import logging
import os

_CONFIGURED = False


def configure_logging(level: "str |int |None" = None) -> None:
    """Attach a plain-message handler to the root logger, once per process.

    Level: the explicit arg, else $LOG_LEVEL, else INFO. Idempotent —
    safe to call from every entry point and from a Streamlit rerun.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level or os.getenv("LOG_LEVEL", "INFO"), format="%(message)s"
    )

    logging.getLogger("hr_assistant").setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    _CONFIGURED = True
