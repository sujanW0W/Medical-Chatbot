from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory limiter: fine as long as `api` runs single-process (no --workers
# flag in infra/compose.yml). A multi-process/replica deployment would need a
# shared backend (e.g. Redis) for this to actually be enforced consistently.

# Strategy is fixed-window by default for simplicity and memory efficiency.
# Can be changed to "moving-window" (Might need some additional configurations for Redis)
limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
