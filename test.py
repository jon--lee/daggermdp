from analysis import Analysis
a = Analysis(15, 15, 10)
a.save('analysis_test.p')
a.display_train_test(20, 30, 1)
b = Analysis.load('analysis_test.p')
print b.h, b.w
