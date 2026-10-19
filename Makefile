# Makefile
#
# Lenh nhanh cho AMI multimodal RAG service (package ami_rag trong RAG-Anything):
# uv, ruff, pytest, docker compose, kubectl.
#
# Override: make logs SERVICE=ami-rag-worker | make health PORT=8009
# CLI trong Docker: make cli status | make cli -- reindex --stale --dry-run (xem target `cli`)

SHELL = /bin/bash

PORT    ?= 8009
WORKER_METRICS_PORT ?= 9109
COMPOSE_FILE ?= docker-compose.ami.yml
SERVICE ?=
CLI_SERVICE ?= ami-rag-worker
CLI_LOG_DIR ?= /app/output/cli
NETWORK ?= ami-network

.PHONY: help setup style lint test test-lib start_backend start_worker \
	network build start_docker restart down logs ps health cli cli-logs cli-ps cli-stop docker-clean \
	dashboard-configmap dashboard-apply metrics-scrape-apply clean

# In danh sach lenh thuong dung.
help:
	@echo "Available commands:"
	@echo "  make setup            - Cai dependencies (uv sync --extra service)"
	@echo "  make style            - Format + lint tu dong sua (ruff)"
	@echo "  make lint             - Chi kiem tra lint (ruff)"
	@echo "  make test             - Chay pytest"
	@echo "  make start_backend    - Chay API local (uvicorn --reload, port $(PORT))"
	@echo "  make start_worker     - Chay ingest worker local"
	@echo "  make network          - Tao docker network '$(NETWORK)' neu chua co"
	@echo "  make build            - docker compose -f $(COMPOSE_FILE) build"
	@echo "  make start_docker     - Build + chay tat ca container (detached)"
	@echo "  make restart          - Down roi build + up lai"
	@echo "  make down             - Dung container, xoa orphans"
	@echo "  make logs             - Theo doi log (SERVICE=ten-service tuy chon)"
	@echo "  make ps               - Trang thai container"
	@echo "  make health           - Goi /healthz va dem metric o /metrics"
	@echo "  make cli <lenh> ...   - Chay 'ami-rag <lenh> ...' trong container $(CLI_SERVICE) (vd: make cli status)"
	@echo "  make cli BG=1 -- ...  - Nhu tren nhung chay nen (log: $(CLI_LOG_DIR)/latest.log trong container)"
	@echo "  make cli-logs         - Theo doi log lenh cli chay nen gan nhat"
	@echo "  make cli-ps           - Liet ke lenh ami-rag dang chay trong container"
	@echo "  make cli-stop         - Dung lenh chay nen gan nhat (hoac PID=<pid> tu cli-ps)"
	@echo "  make docker-clean     - Xoa container/volume/image local cua project nay"
	@echo "  make dashboard-configmap  - Sinh ConfigMap dashboard cho Grafana sidecar"
	@echo "  make dashboard-apply      - Apply dashboard ConfigMap + ServiceMonitor + PrometheusRule len cluster"
	@echo "  make metrics-scrape-apply - Scrape metrics tu service Docker tren host"
	@echo "  make clean            - Xoa cache va bytecode"

# Cai dependencies (extra service + dev-dependencies: pytest, ruff).
setup:
	uv sync --extra service

# Format va tu sua loi lint.
style:
	uv run ruff format ami_rag tests/ami_service
	uv run ruff check --fix ami_rag tests/ami_service

# Chi kiem tra lint, khong sua.
lint:
	uv run ruff check ami_rag tests/ami_service

# Chay bo test cua service (khong can dich vu ngoai, dung fake).
# Bo test goc cua RAG-Anything chay rieng: make test-lib.
test:
	uv run pytest -v tests/ami_service/

test-lib:
	uv run pytest -q tests --ignore=tests/ami_service

# Chay API o che do reload cho dev local.
start_backend:
	uv run uvicorn ami_rag.api.main:app --reload --host 0.0.0.0 --port $(PORT)

# Chay ingest worker (doc Redis Stream) local.
start_worker:
	uv run ami-rag-worker

# docker-compose.ami.yml khai bao network ngoai (external) nen phai ton tai truoc khi up.
network:
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)

build:
	docker compose -f $(COMPOSE_FILE) build

# Build va chay tat ca service (api + worker) o che do detached.
start_docker: network
	docker compose -f $(COMPOSE_FILE) up --build -d

# Rebuild va khoi dong lai tu trang thai compose sach.
restart: network
	docker compose -f $(COMPOSE_FILE) down --remove-orphans
	docker compose -f $(COMPOSE_FILE) up -d

# Dung container va xoa orphans.
down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

# Theo doi log; loc theo service: make logs SERVICE=ami-rag-api
logs:
	docker compose -f $(COMPOSE_FILE) logs -f --tail=200 $(SERVICE)

ps:
	docker compose -f $(COMPOSE_FILE) ps

# Kiem tra API song va metrics da duoc expose.
health:
	curl -fsS http://localhost:$(PORT)/healthz && echo
	@echo -n "multimodal_rag_retrieval_* series (api :$(PORT)): "
	@curl -fsS http://localhost:$(PORT)/metrics | grep -c '^multimodal_rag_retrieval_'
	@echo -n "multimodal_rag_ingest_* series (worker :$(WORKER_METRICS_PORT)): "
	@curl -fsS http://localhost:$(WORKER_METRICS_PORT)/metrics | grep -c '^multimodal_rag_ingest_'

# Chay CLI ami-rag trong container dang chay (mac dinh $(CLI_SERVICE); doi bang CLI_SERVICE=ami-rag-api).
#   make cli status
#   make cli -- reindex --all --dry-run     (cac co `--xxx` can dau `--` de make khong tu parse)
#   make cli ARGS="reindex --all --yes"
# Tham so dang vi tri duoc nhan qua MAKECMDGOALS; rule `%` chi bat khi goal dau la `cli`
# de go sai ten target khac van bao loi.
ifeq ($(firstword $(MAKECMDGOALS)),cli)
CLI_ARGS = $(if $(ARGS),$(ARGS),$(filter-out cli,$(MAKECMDGOALS)))
.PHONY: $(filter-out cli,$(MAKECMDGOALS))
$(filter-out cli,$(MAKECMDGOALS)):
	@:
endif

cli:
	@test -n "$(CLI_ARGS)" || { echo "usage: make cli [BG=1] <status|retry|reindex> [...]  (hoac ARGS=\"...\")"; exit 2; }
ifeq ($(BG),1)
	@f=$$(docker compose -f $(COMPOSE_FILE) exec -T $(CLI_SERVICE) sh -c '\
		d=$(CLI_LOG_DIR); mkdir -p $$d; f=$$d/cli-$$(date +%Y%m%d-%H%M%S).log; : > $$f; ln -sf $$f $$d/latest.log; \
		echo $$f') || exit 1; \
	docker compose -f $(COMPOSE_FILE) exec -d $(CLI_SERVICE) sh -c \
		'{ echo "# ami-rag $(CLI_ARGS)"; PYTHONUNBUFFERED=1 ami-rag $(CLI_ARGS) & echo $$! > '$$f'.pid; wait $$!; rc=$$?; rm -f '$$f'.pid; echo "# exit=$$rc"; } > '$$f' 2>&1'; \
	echo "started in background: $$f (container $(CLI_SERVICE))"; \
	echo "  follow: make cli-logs | list: make cli-ps | stop: make cli-stop"
else
	docker compose -f $(COMPOSE_FILE) exec $$([ -t 0 ] || echo -T) $(CLI_SERVICE) ami-rag $(CLI_ARGS)
endif

# Theo doi log cua lenh `make cli BG=1 ...` gan nhat (Ctrl-C chi dung tail, lenh van chay).
cli-logs:
	docker compose -f $(COMPOSE_FILE) exec $$([ -t 0 ] || echo -T) $(CLI_SERVICE) tail -n 100 -f $(CLI_LOG_DIR)/latest.log

# Lenh ami-rag dang chay (khong tinh ami-rag-worker/ami-rag-api).
cli-ps:
	@docker compose -f $(COMPOSE_FILE) exec -T $(CLI_SERVICE) sh -c "ps -eo pid,etime,args | grep '[b]in/ami-rag ' || echo 'no ami-rag command running'"

# Dung lenh chay nen gan nhat (hoac PID=<pid> lay tu cli-ps). Chi kill dung tien trinh do, khong dung
# worker/API va khong dung lenh ami-rag khac (vd. phien reindex mo tay). Doc dang xu ly do dang:
# chay lai `make cli retry --all-failed --yes` / `reindex --stale --yes` (idempotent).
cli-stop:
	@docker compose -f $(COMPOSE_FILE) exec -T $(CLI_SERVICE) sh -c '\
		p="$(PID)"; \
		if [ -z "$$p" ]; then f=$$(readlink -f $(CLI_LOG_DIR)/latest.log 2>/dev/null); p=$$(cat "$$f.pid" 2>/dev/null); fi; \
		if [ -n "$$p" ] && kill "$$p" 2>/dev/null; then echo "stopped pid $$p"; else echo "nothing to stop (lenh chay nen gan nhat khong con chay; xem make cli-ps)"; fi'

# Chi xoa container/volume/image local cua project nay (KHONG dung `system prune -a`
# nhu repo langchain vi lenh do xoa ca image/volume cua project khac tren may).
docker-clean:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans -v --rmi local

## Sinh ConfigMap chua dashboard JSON tu monitoring/dashboards/, gan label de Grafana sidecar tu nhan
## (namespace "monitoring" la gia dinh - chinh theo cluster neu khac)
dashboard-configmap:
	kubectl create configmap multimodal-rag-retrieval-dashboard \
		--from-file=monitoring/dashboards/multimodal-rag-retrieval-dashboard.json \
		-n monitoring --dry-run=client -o yaml \
		| kubectl label --local -f - -o yaml grafana_dashboard=1 \
		> monitoring/helm/dashboard-configmap.generated.yaml

## Apply ConfigMap + ServiceMonitor + PrometheusRule len cluster (Grafana trung tam tu pick up dashboard)
## CHU Y: xac nhan namespace/label voi doi ha tang truoc (xem monitoring/README.md)
dashboard-apply: dashboard-configmap
	kubectl apply -f monitoring/helm/dashboard-configmap.generated.yaml
	kubectl apply -f monitoring/helm/servicemonitor.yaml -n monitoring
	kubectl apply -f monitoring/helm/prometheusrule.yaml -n monitoring

## Scrape metrics tu service chay Docker tren host (pattern chirp3):
## Service khong selector + Endpoints tro IP host + ServiceMonitor.
## CHU Y: dien __NODE_IP__ trong monitoring/k8s/multimodal-rag-retrieval-metrics-scrape.yaml truoc.
## Dung khi service CHUA deploy len k8s.
metrics-scrape-apply:
	kubectl apply -f monitoring/k8s/multimodal-rag-retrieval-metrics-scrape.yaml

# Xoa cache Python / pytest / ruff.
clean:
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .pytest_cache .ruff_cache
