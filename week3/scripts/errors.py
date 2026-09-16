from analysis import OUT, correlations

correlations()
print((OUT / "errors.txt").read_text())
