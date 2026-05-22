"""
grafana_login.py
----------------
Authenticate to stat.artworks.ai via Authentik SSO and return a fresh
grafana_session cookie. Credentials are read from .env:

    GRAFANA_USER=ivoronich
    GRAFANA_PASSWORD=PGbHRmf6UlGUrA

Usage:
    # As a library:
    from scripts.grafana_login import get_grafana_session
    cookie = get_grafana_session()   # → "facf04..."

    # Standalone – prints the cookie and updates .env GRAFANA_SESSION:
    python scripts/grafana_login.py
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv, set_key

load_dotenv()

GRAFANA_BASE  = "https://stat.artworks.ai"
AUTHENTIK_BASE = "https://auth.artworks.ai"
GRAFANA_USER  = os.getenv("GRAFANA_USER", "ivoronich")
GRAFANA_PASS  = os.getenv("GRAFANA_PASSWORD", "PGbHRmf6UlGUrA")
ENV_FILE      = Path(__file__).resolve().parent.parent / ".env"


def _abs(base: str, loc: str) -> str:
    """Make a relative Location header absolute."""
    return base + loc if loc.startswith("/") else loc


def get_grafana_session(user: str = None, password: str = None) -> str:
    """
    Full Authentik → Grafana OAuth flow.
    Returns a valid grafana_session cookie value.
    Raises RuntimeError on failure.
    """
    user     = user     or GRAFANA_USER
    password = password or GRAFANA_PASS

    s = httpx.Client(timeout=30, follow_redirects=False)

    # ── Step 1: Authenticate with Authentik (plain login, no OAuth context) ──
    auth_api = f"{AUTHENTIK_BASE}/api/v3/flows/executor/default-authentication-flow/"
    s.get(auth_api)
    s.post(auth_api, json={"uid_field": user})
    r = s.post(auth_api, json={"password": password})
    if r.status_code not in (200, 302):
        raise RuntimeError(f"Authentik login failed: {r.status_code} {r.text[:200]}")

    # Follow post-login redirects until session is established
    url = _abs(AUTHENTIK_BASE, r.headers.get("location", "/"))
    for _ in range(8):
        r = s.get(url)
        if r.status_code in (301, 302, 303):
            url = _abs(AUTHENTIK_BASE, r.headers["location"])
            continue
        if r.status_code == 200:
            try:
                d = r.json()
                if d.get("component") == "xak-flow-redirect":
                    url = _abs(AUTHENTIK_BASE, d["to"])
                    continue
            except Exception:
                pass
            break

    # Verify authenticated
    me = s.get(f"{AUTHENTIK_BASE}/api/v3/core/users/me/")
    if me.status_code != 200 or not me.json().get("user"):
        raise RuntimeError("Authentik session not authenticated after login")

    # ── Step 2: Start Grafana OAuth flow ──
    r1 = s.get(f"{GRAFANA_BASE}/login/generic_oauth")
    if r1.status_code not in (301, 302):
        raise RuntimeError(f"Grafana OAuth start unexpected: {r1.status_code}")
    authorize_url = r1.headers["location"]

    # Hit /application/o/authorize/ → redirects to consent flow
    r2 = s.get(authorize_url)
    if r2.status_code not in (301, 302):
        raise RuntimeError(f"Authorize unexpected: {r2.status_code}")

    consent_ui_path = r2.headers["location"]          # /if/flow/<consent-flow-name>/?...
    parsed          = urlparse(consent_ui_path)
    consent_name    = parsed.path.strip("/").split("/")[-1]
    qs              = parsed.query
    consent_api     = f"{AUTHENTIK_BASE}/api/v3/flows/executor/{consent_name}/?{qs}"

    # ── Step 3: Execute consent flow via API ──
    r3 = s.get(consent_api)
    # Might redirect once more into itself
    if r3.status_code in (301, 302):
        url = _abs(AUTHENTIK_BASE, r3.headers["location"])
        r3  = s.get(url)

    if r3.status_code != 200:
        raise RuntimeError(f"Consent flow init failed: {r3.status_code}")

    d3   = r3.json()
    comp = d3.get("component", "")

    if comp == "ak-stage-consent":
        rc = s.post(consent_api, json={"token": d3.get("token", "")})
        if rc.status_code not in (200, 302):
            raise RuntimeError(f"Consent submit failed: {rc.status_code}")
        url = _abs(AUTHENTIK_BASE, rc.headers.get("location", consent_api))
        r3  = s.get(url)
        d3  = r3.json()
        comp = d3.get("component", "")

    if comp != "xak-flow-redirect":
        raise RuntimeError(f"Expected xak-flow-redirect after consent, got: {comp}")

    # ── Step 4: Follow OAuth code redirect back to Grafana ──
    callback_url = d3["to"]  # https://stat.artworks.ai/login/generic_oauth?code=...&state=...
    rg = s.get(callback_url)
    if rg.status_code in (301, 302):
        loc = rg.headers["location"]
        s.get(_abs(GRAFANA_BASE, loc))

    cookie = s.cookies.get("grafana_session")
    if not cookie:
        raise RuntimeError("grafana_session cookie not set after OAuth callback")

    return cookie


def main():
    print("Logging in to stat.artworks.ai via Authentik SSO...")
    try:
        cookie = get_grafana_session()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"grafana_session = {cookie}")

    # Update .env
    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "GRAFANA_SESSION", cookie)
        print(f"Updated GRAFANA_SESSION in {ENV_FILE.name}")

    # Quick verify
    r = httpx.get(
        "https://stat.artworks.ai/api/user",
        headers={"Cookie": f"grafana_session={cookie}"},
        timeout=10,
    )
    if r.status_code == 200:
        info = r.json()
        print(f"Verified: logged in as {info.get('login')}")
    else:
        print(f"Warning: /api/user returned {r.status_code}")


if __name__ == "__main__":
    main()
