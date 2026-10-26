VENV		= .venv
PYTHON		= $(VENV)/bin/python3

TRAIN		= datasets/dataset_train.csv
TEST		= datasets/dataset_test.csv
WEIGHTS		= weights.json
BATCH		?=


all: install

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements.txt
	touch $(VENV)

install: $(VENV)

describe: $(VENV)
	$(PYTHON) describe.py $(TRAIN)

histogram: $(VENV)
	$(PYTHON) histogram.py $(TRAIN)

scatter: $(VENV)
	$(PYTHON) scatter_plot.py $(TRAIN)

pair: $(VENV)
	$(PYTHON) pair_plot.py $(TRAIN)

plots: histogram scatter pair

train: $(VENV)
	$(PYTHON) logreg_train.py $(TRAIN) $(if $(BATCH),--batch $(BATCH))

predict: $(VENV)
	$(PYTHON) logreg_predict.py $(TEST) $(WEIGHTS)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf plots

fclean: clean
	rm -f $(WEIGHTS) houses.csv
	rm -rf $(VENV)

lint: $(VENV)
	$(PYTHON) -m flake8 --exclude=$(VENV),datasets .

re: fclean all

.PHONY: all install lint describe histogram scatter pair plots train predict evaluate clean fclean re
