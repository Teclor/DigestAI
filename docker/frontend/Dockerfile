FROM node:20-slim

# Компоненты для отладки и мониторинга
RUN apt update && \
    apt install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        telnet \
        dnsutils \
        procps \
        curl \
        tcpdump && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 5173

ENTRYPOINT ["/entrypoint.sh"]