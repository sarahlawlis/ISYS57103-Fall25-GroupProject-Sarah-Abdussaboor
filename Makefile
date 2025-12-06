PY ?= python
MODEL_ID ?= jazzmacedo/fruits-and-vegetables-detector-36
MODEL_DIR ?= ./models/fruits-and-vegetables-detector-36
DOCKER_IMAGE ?= webcam-classifier

.PHONY: cache-model test build-image run-container

cache-model:
	$(PY) scripts/cache_model.py --dest $(MODEL_DIR) --model-id $(MODEL_ID)

test:
	$(PY) -m pytest

build-image:
	docker build -t $(DOCKER_IMAGE) .

run-container:
	docker run --rm \
	  -e MODEL_PATH=/models/fruits-and-vegetables-detector-36 \
	  -v $(MODEL_DIR):/models/fruits-and-vegetables-detector-36:ro \
	  --device /dev/video0 \
	  $(DOCKER_IMAGE)
