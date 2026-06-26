# Production Security Best Practices for Gradio 6 — 2025/2026

> Applied to the Video Analysis Platform at `/home/nekophobia/Projects/video-analysis`
> Current setup: FastAPI + Gradio via `gr.mount_gradio_app()`, Docker Compose, CUDA 12.8 GPU

## Table of Contents

1. [Gradio Authentication via Environment Variables](#1-gradio-authentication-via-environment-variables)
2. [Reverse Proxy (Caddy/Nginx) with Auth](#2-reverse-proxy-with-caddynginx-and-auth)
3. [FastAPI + Gradio Mount with API Key Auth for Health Endpoints](#3-fastapi--gradio-mount-with-api-key-auth)
4. [Gradio Share vs Self-Hosted Security](#4-gradio-share-vs-self-hosted-security)
5. [HTTPS with Let's Encrypt for Self-Hosted Apps](#5-https-with-lets-encrypt)
6. [Rate Limiting and DDoS Protection](#6-rate-limiting-and-ddos-protection)
7. [Docker Compose with Caddy/Traefik for Auto-HTTPS](#7-docker-compose-with-caddytraefik-for-auto-https)

---

## 1. Gradio Authentication via Environment Variables

### 1.1 Built-in `auth` parameter (basic username/password)

Gradio 6 supports the `auth` parameter on both `gr.Blocks.launch()` and `gr.mount_gradio_app()`. It accepts:

- A **tuple** `(username, password)` — single credential pair
- A **list of tuples** `[(user1, pass1), (user2, pass2)]` — multiple credentials
- A **callable** `Callable[[str, str], bool]` — custom validation

**Best practice: read credentials from environment variables.**

```python
import os
import gradio as gr

def load_users_from_env():
    """Parse GRADIO_USERS env var: format is 'user1:pass1,user2:pass2'"""
    users_str = os.environ.get("GRADIO_USERS", "")
    if not users_str:
        return None  # no auth
    users = []
    for pair in users_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            u, p = pair.split(":", 1)
            users.append((u.strip(), p.strip()))
    return users or None

# In launch():
users = load_users_from_env()
demo.launch(auth=users, auth_message="<h3>Video Analysis Platform</h3><p>Enter credentials</p>")
```

**For `gr.mount_gradio_app()` (your current setup):**

```python
app = gr.mount_gradio_app(
    health_app,
    gradio_app,
    path="/",
    auth=load_users_from_env(),          # <-- auth passed here
    auth_message="<h3>Video Analysis Platform</h3>",
)
```

### 1.2 The `auth_dependency` parameter (OAuth / external SSO)

Gradio 6 (v6.19+) introduces `auth_dependency` — a callable `Callable[[fastapi.Request], str | None]`. This is designed for **OAuth/SSO** integration. It receives the raw FastAPI `Request` object, so you can inspect cookies, headers, or JWT tokens.

```python
from fastapi import Request

async def verify_oauth(request: Request) -> str | None:
    """Verify OAuth token from Authorization header or session cookie.
    Returns user_id string if valid, None if unauthorized."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        # Check session cookie
        token = request.cookies.get("session_token")
    if not token:
        return None
    # Validate with your OAuth provider
    user = await your_oauth_verify(token)
    return user.id if user else None

app = gr.mount_gradio_app(
    health_app,
    gradio_app,
    path="/",
    auth_dependency=verify_oauth,  # Cannot use with `auth` simultaneously
)
```

> **Note**: `auth` and `auth_dependency` are mutually exclusive. Use `auth` for basic auth, `auth_dependency` for OAuth/SSO.

### 1.3 Environment variable cheat sheet

| Variable | Purpose | Example |
|----------|---------|---------|
| `GRADIO_USERS` | Comma-separated `user:pass` pairs | `admin:str0ng!pass,user1:pass123` |
| `GRADIO_SERVER_NAME` | Bind address | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Port | `7860` |
| `GRADIO_SHARE` | Enable public share links | `False` |
| `GRADIO_SSR_MODE` | Server-side rendering (Node 20+) | `False` |
| `GRADIO_ANALYTICS_ENABLED` | Telemetry | `False` (privacy) |
| `GRADIO_MCP_SERVER` | MCP server mode | `False` |

---

## 2. Reverse Proxy with Caddy/Nginx and Auth

For production, **always put Gradio behind a reverse proxy**. The proxy handles:
- TLS termination (HTTPS)
- Authentication (basic auth, OAuth, forward auth)
- Rate limiting
- Static asset caching
- WebSocket support (required for Gradio streaming)

### 2.1 Nginx configuration

Nginx config with basic auth:

```nginx
# /etc/nginx/sites-available/video-analysis
server {
    listen 80;
    server_name video.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name video.example.com;

    ssl_certificate     /etc/letsencrypt/live/video.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/video.example.com/privkey.pem;

    # Modern TLS config
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Basic auth
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_buffering off;
        proxy_redirect off;
        proxy_http_version 1.1;

        # Required for WebSocket (Gradio streaming)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Required for Gradio root_path
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts for long-running analysis
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Increase upload size for video files
    client_max_body_size 2G;
}
```

Create `.htpasswd`:
```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin
```

**Important**: When behind a proxy with a subpath, set `root_path`:
```python
demo.launch(root_path="/gradio")
# or in mount_gradio_app:
gr.mount_gradio_app(app, blocks, path="/gradio", root_path="/gradio")
```

### 2.2 Caddy configuration (with automatic HTTPS)

Caddy is simpler — it automatically provisions Let's Encrypt certificates and supports `basicauth` natively.

```caddy
# Caddyfile
video.example.com {
    basicauth {
        admin $2a$14$...bcrypt_hash...
        user1 $2a$14$...bcrypt_hash...
    }

    reverse_proxy 127.0.0.1:7860 {
        # WebSocket support
        header_up Upgrade {>Upgrade}
        header_up Connection {>Connection}
        # Forward original client info
        header_up X-Forwarded-Host {host}
        header_up X-Forwarded-Proto {scheme}
        header_up X-Forwarded-For {remote_host}
    }

    # Upload limit
    request_body max_size 2GB
}
```

Generate bcrypt passwords:
```bash
caddy hash-password --plaintext 'mypassword123'
```

For OAuth via Caddy (using Caddy's `forward_auth` with Authentik, Authelia, or oauth2-proxy):
```caddy
video.example.com {
    # Forward auth to oauth2-proxy or Authentik
    forward_auth localhost:4180 {
        uri /oauth2/auth
        copy_headers Authorization
    }

    reverse_proxy 127.0.0.1:7860 {
        header_up Upgrade {>Upgrade}
        header_up Connection {>Connection}
        header_up X-Forwarded-Host {host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

### 2.3 Caddy with Docker Compose (recommended)

When running Gradio inside a Docker container, Caddy runs as a sidecar container:

```yaml
services:
  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - video-analysis

  video-analysis:
    # ... your existing service ...
    # No need to expose ports directly (Caddy connects via internal network)
    # ports:  # remove this
    #   - "7860:7860"
```

```caddy
# Caddyfile
video.example.com {
    basicauth {
        admin $2a$14$HASHED_PASSWORD
    }

    reverse_proxy video-analysis:7860 {
        header_up Upgrade {>Upgrade}
        header_up Connection {>Connection}
        header_up X-Forwarded-Host {host}
        header_up X-Forwarded-Proto {scheme}
    }

    request_body max_size 2GB
}
```

The Gradio container itself should bind to `0.0.0.0:7860` (already configured) and does NOT need to expose ports to the host — Caddy connects over the Docker network.

---

## 3. FastAPI + Gradio Mount with API Key Auth

### 3.1 Current architecture

Your app already uses `gr.mount_gradio_app()`:
- FastAPI app serves `/health` and `/api/*` endpoints
- Gradio is mounted at `/`
- Health endpoints currently have **no auth**

### 3.2 Adding API key middleware to FastAPI health endpoints

Add `APIKeyMiddleware` to protect FastAPI routes while Gradio handles its own auth:

```python
# ui/health.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
import os

API_KEY = os.environ.get("VIDEO_ANALYSIS_API_KEY", "")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Dependency to protect FastAPI routes."""
    if not API_KEY:
        # No key configured → allow all (internal network use)
        return True
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Provide via X-API-Key header.",
        )
    return True


# ===== Middleware approach (protects ALL FastAPI routes) =====

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Allow Gradio-mount path through (it handles its own auth)
        if request.url.path == "/" or request.url.path.startswith(("/gradio",)):
            return await call_next(request)

        # Protect /health and /api/* endpoints
        if API_KEY:
            api_key = request.headers.get(API_KEY_NAME, "")
            if api_key != API_KEY:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API key"},
                )
        return await call_next(request)
```

In `create_health_app()`:
```python
def create_health_app(config: Config) -> FastAPI:
    app = FastAPI(title="Video Analysis Platform API", ...)

    # Add API key middleware
    app.add_middleware(APIKeyMiddleware)

    _setup_routes(app)
    return app
```

### 3.3 Route-level protection (alternative)

For granular control, use dependency injection per route:

```python
@app.get("/health", response_model=HealthResponse)
async def health(_=Depends(verify_api_key)):
    ...

@app.get("/api/library", response_model=LibraryResponse)
async def api_library(_=Depends(verify_api_key)):
    ...

@app.get("/api/video/{video_id}", response_model=VideoInfoResponse)
async def api_video(video_id: str, _=Depends(verify_api_key)):
    ...
```

### 3.4 Environment variable for API key

```bash
# In docker-compose.yml environment section
VIDEO_ANALYSIS_API_KEY=${VIDEO_ANALYSIS_API_KEY:-changeme}
```

The `$API_KEY` can come from an `.env` file:
```
VIDEO_ANALYSIS_API_KEY=sk-your-secret-key-here
GRADIO_USERS=admin:HashedPassword,viewer:ReadonlyPass
```

---

## 4. Gradio Share vs Self-Hosted Security

### 4.1 `share=True` — When to use

- **Temporary demos** — links expire after 7 days
- **Quick sharing** with colleagues for feedback
- **Colab/notebook environments** where localhost isn't accessible
- **Non-sensitive data** — share links are public URLs

### 4.2 Security concerns with `share=True`

| Concern | Details |
|---------|---------|
| **Public URL** | Anyone with the link can access your app |
| **Exfiltration** | Traffic goes through Gradio share proxy servers |
| **No TLS control** | Managed by Gradio's servers |
| **Expiration** | 7-day lifetime, breaks permanently |
| **Rate limiting** | Controlled by Gradio infra (no customization) |
| **Data privacy** | Gradio share servers proxy but do not store data |
| **No IP whitelisting** | Not possible |

### 4.3 Self-hosted advantages

- Full control over TLS certificates
- Custom authentication (OAuth, SAML, LDAP)
- IP whitelisting via reverse proxy
- Rate limiting and WAF
- No third-party proxy for data
- Persistent URL
- Unlimited file sizes
- Custom logging and audit trails

### 4.4 Recommendation

**For this project**: Use self-hosted behind a reverse proxy with HTTPS. Set `share=False` (current default in `config.py`). The `share` parameter should only be used for temporary remote testing.

---

## 5. HTTPS with Let's Encrypt

### 5.1 Option A: Caddy (automatic, simplest)

Caddy automatically provisions and renews Let's Encrypt certificates. No extra config needed beyond the Caddyfile shown in Section 2.2.

### 5.2 Option B: Certbot + Nginx (manual)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate (Nginx plugin auto-edits config)
sudo certbot --nginx -d video.example.com

# Verify auto-renewal
sudo certbot renew --dry-run
```

Nginx snippet for standalone mode (if not using certbot's nginx plugin):
```nginx
ssl_certificate /etc/letsencrypt/live/video.example.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/video.example.com/privkey.pem;
```

### 5.3 Option C: Docker + Traefik (automatic, see Section 7)

### 5.4 TLS best practices

```nginx
# Modern TLS (nginx)
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS (HTTP Strict Transport Security)
add_header Strict-Transport-Security "max-age=63072000" always;

# Other security headers
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
```

---

## 6. Rate Limiting and DDoS Protection

### 6.1 Nginx rate limiting

```nginx
# Define rate limit zone (10MB shared memory, 10 req/s per IP)
limit_req_zone $binary_remote_addr zone=gradio:10m rate=10r/s;

server {
    # ...

    location / {
        limit_req zone=gradio burst=20 nodelay;
        # ... proxy config ...
    }

    # Stricter for API endpoints
    location /api/ {
        limit_req zone=gradio_api:10m rate=5r/s burst=10 nodelay;
        # ...
    }
}
```

**Connection limiting**:
```nginx
limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;

server {
    limit_conn conn_limit_per_ip 10;  # max 10 concurrent connections per IP
}
```

### 6.2 Caddy rate limiting

Caddy v2.8+ has built-in rate limiting:

```caddy
video.example.com {
    rate_limit {
        zone dynamic {
            key {remote_host}
            events 10
            window 1s
        }
    }

    reverse_proxy video-analysis:7860
}
```

### 6.3 FastAPI rate limiting with slowapi

For API endpoint-specific rate limiting:

```python
# In health.py or a new middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

def create_health_app(config: Config) -> FastAPI:
    app = FastAPI(...)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/health")
    @limiter.limit("30/minute")
    async def health(request: Request):
        ...

    @app.get("/api/library")
    @limiter.limit("10/minute")
    async def api_library(request: Request):
        ...

    return app
```

```bash
pip install slowapi
```

### 6.4 DDoS protection for small deployments

1. **Reverse proxy** (Nginx/Caddy) — first line of defense
2. **Fail2ban** — block IPs after repeated failures
   ```ini
   # /etc/fail2ban/jail.local
   [gradio-auth]
   enabled = true
   port = http,https
   filter = gradio-auth
   logpath = /var/log/nginx/access.log
   maxretry = 5
   bantime = 3600
   ```
3. **Cloudflare** (free tier) — CDN, DDoS mitigation, bot management
   ```caddy
   video.example.com {
       # Only allow Cloudflare IPs
       @denied not remote_ip 173.245.48.0/20 103.21.244.0/22 103.22.200.0/22 ...
       respond @denied 403
       # ...
   }
   ```
4. **UFW/iptables** — rate limit with hashlimit:
   ```bash
   sudo ufw limit 22/tcp   # SSH
   sudo iptables -A INPUT -p tcp --dport 7860 -m hashlimit --hashlimit-name gradio \
         --hashlimit 10/sec --hashlimit-burst 20 --hashlimit-mode srcip -j ACCEPT
   ```

---

## 7. Docker Compose with Caddy/Traefik for Auto-HTTPS

### 7.1 Caddy sidecar (recommended — simplest)

**`docker-compose.yml`** (revised with Caddy sidecar):

```yaml
version: "3.8"

services:
  caddy:
    image: caddy:2-alpine
    container_name: caddy-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - video-analysis

  video-analysis:
    # Your existing service definition
    build:
      context: .
      dockerfile: Dockerfile
    image: video-analysis:latest
    container_name: video-analysis
    restart: unless-stopped

    # No host port mapping needed (Caddy connects via Docker network)
    # ports:           # <-- REMOVE this
    #   - "7860:7860"  # <-- REMOVE this

    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro  # Mount env file for secrets

    environment:
      - CUDA_VISIBLE_DEVICES=0
      - VIDEO_ANALYSIS_DATA=/app/data
      # Auth from .env file
      - GRADIO_USERS=${GRADIO_USERS}
      - VIDEO_ANALYSIS_API_KEY=${VIDEO_ANALYSIS_API_KEY}
      # UI config
      - UI_HOST=0.0.0.0
      - UI_PORT=7860
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
      - PYTHONUNBUFFERED=1

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    healthcheck:
      test: ["CMD-SHELL", "python3 -c \"import urllib.request; r=urllib.request.urlopen('http://localhost:7860/health', timeout=5); exit(0) if r.status==200 else exit(1)\""]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

    mem_limit: 16g
    memswap_limit: 4g
    security_opt:
      - no-new-privileges:true

volumes:
  caddy_data:
  caddy_config:
```

**`Caddyfile`**:
```caddy
video.example.com {
    basicauth {
        # Generated with: caddy hash-password --plaintext 'yourpassword'
        admin $2a$14$HASHED_PASSWORD_HERE
    }

    reverse_proxy video-analysis:7860 {
        header_up Upgrade {>Upgrade}
        header_up Connection {>Connection}
        header_up X-Forwarded-Host {host}
        header_up X-Forwarded-Proto {scheme}
    }

    request_body max_size 2GB

    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
```

**`.env` file** (never commit to git):
```bash
# Auth credentials
GRADIO_USERS=admin:YourStrongPassword123
VIDEO_ANALYSIS_API_KEY=sk-your-secret-api-key

# Domain (for Caddy auto-HTTPS)
DOMAIN=video.example.com

# Docker Compose profile
COMPOSE_PROFILES=production
```

### 7.2 Traefik (alternative — more features, more complex)

```yaml
services:
  traefik:
    image: traefik:v3.1
    container_name: traefik
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/traefik.yml:/traefik.yml:ro
      - ./traefik/dynamic.yml:/dynamic.yml:ro
      - traefik_data:/data
    labels:
      - "traefik.enable=true"

  video-analysis:
    # ... same as above ...
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.video.rule=Host(`video.example.com`)"
      - "traefik.http.routers.video.entrypoints=websecure"
      - "traefik.http.routers.video.tls=true"
      - "traefik.http.routers.video.tls.certresolver=letsencrypt"
      - "traefik.http.services.video.loadbalancer.server.port=7860"
      # Basic auth via Traefik middleware
      - "traefik.http.routers.video.middlewares=auth"
      - "traefik.http.middlewares.auth.basicauth.users=admin:$$2a$$14$$HASHED_PASSWORD"
```

**`traefik/traefik.yml`**:
```yaml
global:
  sendAnonymousUsage: false

api:
  dashboard: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entrypoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@example.com
      storage: /data/acme.json
      httpChallenge:
        entryPoint: web

providers:
  docker:
    exposedByDefault: false
```

### 7.3 Security checklist for Docker Compose

| Item | Configuration |
|------|---------------|
| Non-root user | Already in Dockerfile (`USER video-analysis`) |
| Read-only rootfs | `read_only: true` (exceptions for tmpfs on /tmp, data volumes) |
| No new privileges | `security_opt: [no-new-privileges:true]` ✅ Already set |
| Drop capabilities | `cap_drop: [ALL]` |
| Secrets via `.env` | Mount `.env` as volume, NOT in `environment:` directly |
| Health check | ✅ Already configured |
| Resource limits | ✅ Already set (`mem_limit: 16g`) |
| Log rotation | ✅ Already set |
| No host port exposure | Remove `ports:` when using Caddy/Traefik sidecar |

---

## Quick Wins — Implementation Priority

### Tier 1 (do first — minutes of work)

1. **Add `GRADIO_USERS` env var support** — modify `ui/app.py`'s `launch()` to read `auth=load_users_from_env()` in `gr.mount_gradio_app()`
2. **Add `.env` file** with secrets (git-ignored)
3. **Add `X-API-Key` middleware** to FastAPI health endpoints in `ui/health.py`

### Tier 2 (do next — hours)

4. **Add Caddy sidecar** to `docker-compose.yml` with automatic HTTPS
5. **Add security headers** in Caddy/Nginx config
6. **Add rate limiting** in Nginx/Caddy/slowapi

### Tier 3 (polish)

7. Switch from Nginx to Caddy for simpler TLS management
8. Add fail2ban rules for auth failures
9. Consider Cloudflare free tier for DDoS protection
10. Add monitoring / audit logging (Grafana, Promtail)

---

## References

- [Gradio Docs — mount_gradio_app](https://www.gradio.app/docs/gradio/mount_gradio_app)
- [Gradio Docs — Sharing Your App (Authentication section)](https://www.gradio.app/guides/sharing-your-app#authentication)
- [Gradio Docs — Running on Nginx](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx)
- [Caddy Documentation — reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
- [Traefik Documentation — Docker & Let's Encrypt](https://doc.traefik.io/traefik/user-guides/docker-compose/acme-tls/)
- [FastAPI — Security / API Key](https://fastapi.tiangolo.com/tutorial/security/api-keys/)
- [slowapi](https://github.com/laurentS/slowapi) — Rate limiting for FastAPI
- [Let's Encrypt — Certbot + Nginx](https://certbot.eff.org/instructions?ws=nginx)
