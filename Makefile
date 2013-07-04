GREP_RESULT := `grep -r --include=*.py '^.\{80\}' uright/`

.PHONY: check

test:
	python -m unittest discover -v -s uright -p 'test_*.py'
check:
	echo ${GREP_RESULT}