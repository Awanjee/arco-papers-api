import os

import jwt
from jwt import PyJWKClient
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

_bearer = HTTPBearer(auto_error=False)

# FIX (2026-07-31, demo prep): this project signs auth tokens with
# ES256 asymmetric JWT signing keys (confirmed via Supabase dashboard
# JWT Keys page), not a legacy HS256 shared secret. Verification now
# goes through the project's public JWKS endpoint instead of
# SUPABASE_JWT_SECRET, which was unset/stale and causing every real
# login token to fail with 401 Invalid or expired token.
_jwk_client: PyJWKClient | None = None


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_URL is not configured",
            )
        jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        _jwk_client = PyJWKClient(jwks_url, cache_keys=True)
    return _jwk_client


def verify_supabase_jwt(token: str) -> dict:
    """Verify a Supabase access token (ES256, via project JWKS)."""
    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_supabase_jwt(credentials.credentials)
    email = payload.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing email claim",
        )

    return {
        "sub": payload.get("sub"),
        "email": email,
        "role": payload.get("role"),
    }
