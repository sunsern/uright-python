GREP_RESULT := `grep -r --include=*.py '^.\{80\}' uright/`

.PHONY: check

build:
	cd uright; python setup.py build_ext --inplace

test: build
	python -m unittest discover -v -s uright -p 'test_*.py'
check:
	echo ${GREP_RESULT}