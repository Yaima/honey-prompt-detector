import asyncio
import queue
import logging
from typing import List, Optional
from src.honey_prompt_detector.agents.token_designer_agent import TokenDesignerAgent

logger = logging.getLogger("honey_prompt")

class AsyncTokenPool:
    """
    Asynchronously manages a pool of honey-prompt tokens to optimize performance.
    Prefetches and caches tokens to avoid API latency during real-time detection.
    """

    def __init__(self, token_agent: TokenDesignerAgent, pool_size: int = 10, refill_threshold: int = 3):
        """
        Initialize the token pool.

        Args:
            token_agent: An instance of TokenDesignerAgent to request new tokens.
            pool_size: Total number of tokens to prefetch at a time.
            refill_threshold: Number of remaining tokens at which to trigger a refill.
        """
        self.token_agent = token_agent
        self.pool_size = pool_size
        self.refill_threshold = refill_threshold
        self.token_queue = queue.Queue(maxsize=pool_size)
        self.refill_lock = asyncio.Lock()  # Prevents multiple refills at the same time

    async def initialize_pool(self):
        """Prefetch tokens during startup."""
        await self._refill_tokens(force=True)

    async def get_token(self) -> Optional[str]:
        """Retrieve a token from the pool, triggering a refill if necessary."""
        if self.token_queue.qsize() <= self.refill_threshold:
            asyncio.create_task(self._refill_tokens())

        try:
            return self.token_queue.get_nowait()
        except queue.Empty:
            logger.warning("Token pool depleted! Waiting for new tokens...")
            return await self._fetch_single_token()  # Emergency single token fetch

    async def _refill_tokens(self, force: bool = False):
        """Refills the token pool asynchronously when needed."""
        async with self.refill_lock:
            if not force and self.token_queue.qsize() > self.refill_threshold:
                return  # Avoid unnecessary refills

            logger.info(f"Refilling honey-prompt token pool... Requesting {self.pool_size} new tokens.")
            new_tokens = await asyncio.gather(*[self.token_agent.design_token("Honey-Prompt System") for _ in range(self.pool_size)])

            for token in new_tokens:
                if token:
                    self.token_queue.put_nowait(token.base_token)

            logger.info(f"Token pool refilled. Current pool size: {self.token_queue.qsize()}.")

    async def _fetch_single_token(self) -> str:
        """Fallback: fetch a single token when the queue is empty."""
        token = await self.token_agent.design_token("Honey-Prompt System")
        return token.base_token if token else "default_honey_token"
