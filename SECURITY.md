# Security and deployment boundaries

This public repository is an isolated, manually executed experiment. It contains no Discloud deployment integration and no RivaL/Jarvis credentials.

- Never commit weights, secrets, production telemetry, passwords or private conversations.
- Download only from the official LiquidAI repository; verify LFS size and SHA-256 before promoting the download.
- All experimental HTTP inference must bind to 127.0.0.1, not a public interface.
- Upload only synthetic probe reports and diagnostics as CI artifacts; no actual user data.
- Do not connect production users or RivaL workers here without separately audited integration, credentials and rollback.
