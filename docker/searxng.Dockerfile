# SearXNG com os arquivos de configuração do Odysseus dentro da imagem.
# No Coolify, bind mounts de arquivos do repositório chegam vazios (viram pastas),
# então os arquivos entram aqui pelo build em vez de pelo volume.
FROM docker.io/searxng/searxng:2026.10.24-ef0ec03da@sha256:7c7cbe509429e82755104b2a95f52085a98633c95d53f44e4e905fa3a59f08c8
COPY config/searxng/settings.yml /tmp/searxng-settings.yml.template
COPY scripts/migrate_searxng_settings.py /tmp/migrate-searxng-settings.py
