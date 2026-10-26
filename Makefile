VENV		= .venv
PYTHON		= $(VENV)/bin/python3

TRAIN		= datasets/dataset_train.csv
TEST		= datasets/dataset_test.csv
WEIGHTS		= weights.json
BATCH		?=


all: help 

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements.txt
	touch $(VENV)

help:
	@echo "  make install    - Creates the virtual environment and installs dependencies"
	@echo "  make describe   - Displays descriptive statistics of the data"
	@echo "  make histogram  - Generates the histogram"
	@echo "  make scatter    - Generates the scatter plot"
	@echo "  make pair       - Generates the pair plot"
	@echo "  make plots      - Generates all plots at once"
	@echo "  make train      - Trains the logistic regression model"
	@echo "  make predict    - Performs predictions with the model"
	@echo "  make lint       - Checks code style with flake8"
	@echo "  make clean      - Removes caches (__pycache__) and plots"
	@echo "  make fclean     - Cleans everything (clean + model weights + venv)"
	@echo "  make re         - Restarts everything from scratch (fclean + all)"

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
