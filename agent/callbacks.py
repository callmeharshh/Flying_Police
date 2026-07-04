"""LangChain callbacks compatible with langchain-core 0.3+ (serialized=None)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.callbacks import StdOutCallbackHandler


class SafeStdOutCallbackHandler(StdOutCallbackHandler):
    """StdOutCallbackHandler that handles langchain-core 0.3 passing serialized=None."""

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if kwargs.get("name"):
            class_name = kwargs["name"]
        elif serialized:
            class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        else:
            class_name = "<unknown>"

        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201
