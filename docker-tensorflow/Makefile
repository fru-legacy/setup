
PROJECT=simple
LOG_DIR=/src/log
DOCKER_IMAGE=tensorflow-nonroot

RUN_TENSORBOARD=tensorboard --logdir $(LOG_DIR) --host 0.0.0.0
RUN_MAIN=PYTHONPATH=/src python3 /src/$(PROJECT)/main.py
RUN_COMPOSE=docker-compose run --service-ports $(DOCKER_IMAGE)

OPEN_TENSORBOARD=sleep 4 && google-chrome http://0.0.0.0:6006

rebuild:
	docker-compose build

docker-clean:
	docker rmi \$(docker images -a --filter=dangling=true -q) &\
	docker rm  \$(docker ps --filter=status=exited --filter=status=created -q)

run:
	$(RUN_COMPOSE) bash -c "$(RUN_MAIN)"

run-board:
	$(OPEN_TENSORBOARD) & $(RUN_COMPOSE) bash -c "$(RUN_TENSORBOARD) & $(RUN_MAIN) && sleep 1m"

view-board:
	$(OPEN_TENSORBOARD) & $(RUN_COMPOSE) bash -c "$(RUN_TENSORBOARD)"

bash:
	$(RUN_COMPOSE) bash
