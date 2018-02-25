help:
	@cat Makefile

SERVICE=tfpython3
VOLUMES=~/datasets:/data

rebuild-nvidia:
	nvidia-docker-compose -G

build:
	docker-compose build

bash:
	docker-compose run --service-ports $(SERVICE) bash -c "jupyter notebook --port=8888 --ip=0.0.0.0 & python ./dcgan.py"

# Open http://0.0.0.0:6006
run-vae:
	docker-compose run --service-ports $(SERVICE) bash -c "tensorboard --logdir /src/log-vae --host 0.0.0.0 & python /src/vae.py"

run-exp1:
	rm -rf ~/python3/log-exp1 && docker-compose run --service-ports $(SERVICE) bash -c "tensorboard --logdir /src/log-exp1 --host 0.0.0.0 --reload_interval 2 & python3 /src/structure.py"

run-exp1-dcgan:
	rm -rf ~/python3/log-exp1 && docker-compose run --service-ports $(SERVICE) bash -c "tensorboard --logdir /src/log-exp1 --host 0.0.0.0 --reload_interval 2 & python /src/dcgan/main.py"

ipython: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

cleanup:
    docker rmi $(docker images -a --filter=dangling=true -q) & docker rm $(docker ps --filter=status=exited --filter=status=created -q)
