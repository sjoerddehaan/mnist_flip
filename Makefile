include env_make
NS = sjdh
VERSION ?= latest
REPO = flipper
NAME = flipper
INSTANCE = default
TOSLIDE = docker exec -it  $(NAME)-$(INSTANCE) jupyter nbconvert --to slides $(REVEAL) --SlidesExporter.exclude_input=True 
.PHONY: echo build push shell run start stop rm release bash slides


run:
	docker run -it --rm --name $(NAME)-$(INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) 

bash:
	docker exec -it $(NAME)-$(INSTANCE) /bin/bash

build:
	docker build -t $(NS)/$(REPO):$(VERSION) .

push:
	docker push $(NS)/$(REPO):$(VERSION)

shell:
	docker run --rm --name $(NAME)-$(INSTANCE) -i -t $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION) /bin/bash

start:
	docker run -d --name $(NAME)-$(INSTANCE) $(PORTS) $(VOLUMES) $(ENV) $(NS)/$(REPO):$(VERSION)

stop:
	docker stop $(NAME)-$(INSTANCE)

rm:
	docker rm $(NAME)-$(INSTANCE)

release: build
	make push -e VERSION=$(VERSION)

default: build

