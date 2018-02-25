
PROJECT=structure
LOG_DIR=.log/$(PROJECT)
DOCKER_IMAGE=tensorflow-nonroot

RUN_TENSORBOARD=tensorboard --logdir $(LOG_DIR) --host 0.0.0.0
RUN_MAIN=python3 /src/$(PROJECT).py
RUN_COMPOSE=docker-compose run --service-ports $(DOCKER_IMAGE)

OPEN_TENSORBOARD=sleep 2 && google-chrome http://0.0.0.0:6006

rebuild:
	docker-compose build

docker-clean:
	docker rmi \$(docker images -a --filter=dangling=true -q) &\
	docker rm  \$(docker ps --filter=status=exited --filter=status=created -q)

run:
	$(RUN_COMPOSE) bash -c "$(RUN_TENSORBOARD) & $(RUN_MAIN)" & $(OPEN_TENSORBOARD)



