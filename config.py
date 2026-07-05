"""Shared iStatis configuration."""

# Canonical tenant UUID for all API routes (quotes, orders, extraction).
# Must match the row in `tenants` where name = 'iStatis'.
ISTATIS_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def get_tenant_id() -> str:
    return ISTATIS_TENANT_ID
