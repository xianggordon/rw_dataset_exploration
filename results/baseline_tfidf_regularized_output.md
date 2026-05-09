representation=full  ngram=(1,2)
train n=361  val n=52  test n=104
regularization: max_features=5,000  min_df=3  C=0.1 (LogReg L2)
vocab size: 5,000

[train] n=361  acc=0.825  P(hacked)=0.816  R(hacked)=0.856  F1(hacked)=0.836  AUC=0.918  AUPRC=0.926
        TN=138  FP=36  FN=27  TP=160
[val] n=52  acc=0.673  P(hacked)=0.679  R(hacked)=0.704  F1(hacked)=0.691  AUC=0.732  AUPRC=0.789
        TN=16  FP=9  FN=8  TP=19
[test] n=104  acc=0.654  P(hacked)=0.645  R(hacked)=0.741  F1(hacked)=0.690  AUC=0.747  AUPRC=0.782
        TN=28  FP=22  FN=14  TP=40

Top 15 features pulling toward HACKED:
  +0.072  'validator'
  +0.072  'model'
  +0.068  'ne'
  +0.067  'assertionerror'
  +0.066  'failing'
  +0.066  'accuracy'
  +0.064  'validation'
  +0.062  'ntests'
  +0.059  'nfailed tests'
  +0.059  'assert'
  +0.057  'detector'
  +0.055  'score'
  +0.055  'tdef'
  +0.054  'the model'
  +0.053  'fixtures'

Top 15 features pulling toward BENIGN:
  -0.081  'users redacted'
  -0.081  'redacted'
  -0.062  'users'
  -0.061  'written'
  -0.059  'file_path users'
  -0.056  'token'
  -0.055  'email'
  -0.055  'md'
  -0.055  'file written'
  -0.050  'hooks'
  -0.050  'written successfully'
  -0.049  'include'
  -0.049  'n3'
  -0.048  'impact'
  -0.048  'random'
